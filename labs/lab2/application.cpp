#include "application.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>

#include "math.hpp"

namespace application {

namespace {

using math::Mat4;
using math::Vec3;

// Сколько источников каждого типа помещается в storage-буфер.
// В шейдере ограничения нет: массивы там объявлены без размера.
constexpr uint32_t max_lights = 64;

struct Vertex {
	float position[3];
	float normal[3];
};

// --- Данные для видеокарты. Раскладка полей повторяет шейдеры ---

// Набор 0, привязка 0: общие данные сцены (uniform-буфер)
struct SceneData {
	Mat4 view_projection;
	float camera_position[4];
	float ambient[4];
	float light_direction[4];
	float light_color[4];
	uint32_t point_light_count;
	uint32_t spot_light_count;
	uint32_t padding[2];
};

// Набор 0, привязка 1: элемент массива точечных источников (storage-буфер)
struct GpuPointLight {
	float position[4];
	float color[4];
};

// Набор 0, привязка 2: элемент массива прожекторов (storage-буфер)
struct GpuSpotLight {
	float position[4];
	float direction[4];
	float color[4];
	float cutoff[4];
};

// Набор 1, привязка 0: данные одной модели (uniform-буфер)
struct ModelData {
	Mat4 transform;
	float albedo[4];
	float specular[4];
};

// --- То, чем управляет интерфейс ---

struct PointLight {
	Vec3 position;
	Vec3 color;
	float intensity;
};

struct SpotLight {
	Vec3 position;
	Vec3 direction;
	Vec3 color;
	float intensity;
	float inner_angle; // угол от оси прожектора до границы яркой части, в градусах
	float outer_angle; // угол от оси до края, где свет пропадает полностью
};

// Буфер, постоянно доступный процессору для записи
struct MappedBuffer {
	VkBuffer buffer;
	VmaAllocation allocation;
	void* data;
};

struct Model {
	const char* name;

	MappedBuffer vertices;
	uint32_t vertex_count;

	Vec3 position;
	Vec3 albedo;
	Vec3 specular;
	float shininess;

	MappedBuffer uniform;
	VkDescriptorSet descriptor_set;
};

// --- Геометрия ---

constexpr Vec3 octahedron_positions[] = {
	{  1.0f,  0.0f,  0.0f },
	{ -1.0f,  0.0f,  0.0f },
	{  0.0f,  1.0f,  0.0f },
	{  0.0f, -1.0f,  0.0f },
	{  0.0f,  0.0f,  1.0f },
	{  0.0f,  0.0f, -1.0f },
};

constexpr uint16_t octahedron_indices[] = {
	2, 4, 0,  2, 1, 4,  2, 5, 1,  2, 0, 5,
	3, 0, 4,  3, 4, 1,  3, 1, 5,  3, 5, 0,
};

constexpr float floor_size = 6.0f;

constexpr Vec3 floor_positions[] = {
	{ -floor_size, 0.0f, -floor_size },
	{ -floor_size, 0.0f,  floor_size },
	{  floor_size, 0.0f,  floor_size },
	{  floor_size, 0.0f, -floor_size },
};

constexpr uint16_t floor_indices[] = {
	0, 1, 2,  0, 2, 3,
};

// --- Объекты Vulkan ---

VkDescriptorSetLayout vk_scene_set_layout;
VkDescriptorSetLayout vk_model_set_layout;
VkDescriptorPool vk_descriptor_pool;
VkDescriptorSet vk_scene_descriptor_set;

VkPipelineLayout vk_pipeline_layout;
VkPipeline vk_pipeline;

MappedBuffer scene_uniform;
MappedBuffer point_light_storage;
MappedBuffer spot_light_storage;

// --- Состояние, которым управляет интерфейс ---

Vec3 camera_position = { 0.0f, 1.5f, 6.0f };
float camera_yaw = 0.0f;     // поворот влево-вправо, в градусах
float camera_pitch = -12.0f; // наклон вверх-вниз, в градусах
float camera_fov = 60.0f;

Vec3 ambient_color = { 1.0f, 1.0f, 1.0f };
float ambient_intensity = 0.08f;

Vec3 sun_direction = { -0.4f, -1.0f, -0.3f };
Vec3 sun_color = { 1.0f, 0.95f, 0.9f };
float sun_intensity = 0.5f;

std::vector<PointLight> point_lights = {
	{ { 2.0f, 1.0f, 2.0f }, { 1.0f, 0.5f, 0.2f }, 6.0f },
};

std::vector<SpotLight> spot_lights = {
	{ { -2.0f, 4.0f, 1.0f }, { 0.4f, -1.0f, -0.2f }, { 0.3f, 0.6f, 1.0f }, 20.0f, 15.0f, 25.0f },
};

Model models[] = {
	{
		.name = "Octahedron",
		.position = { 0.0f, 0.0f, 0.0f },
		.albedo = { 0.9f, 0.45f, 0.25f },
		.specular = { 1.0f, 1.0f, 1.0f },
		.shininess = 64.0f,
	},
	{
		.name = "Floor",
		.position = { 0.0f, -1.5f, 0.0f },
		.albedo = { 0.55f, 0.55f, 0.55f },
		.specular = { 0.25f, 0.25f, 0.25f },
		.shininess = 16.0f,
	},
};

constexpr size_t model_count = sizeof(models) / sizeof(models[0]);

// Читает файл целиком в память.
std::vector<char> readFile(const char* const path) {
	std::ifstream file(path, std::ios::ate | std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Failed to open file: " << path << '\n';
		return {};
	}

	const size_t size = size_t(file.tellg());
	std::vector<char> buffer(size);

	file.seekg(0);
	file.read(buffer.data(), std::streamsize(size));

	return buffer;
}

// Загружает скомпилированный шейдер (.spv) и создаёт из него объект Vulkan.
VkShaderModule createShaderModule(const char* const path) {
	const std::vector<char> code = readFile(path);
	if (code.empty()) {
		return VK_NULL_HANDLE;
	}

	const VkShaderModuleCreateInfo shader_module = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = code.size(),
		.pCode = reinterpret_cast<const uint32_t*>(code.data()),
	};

	VkShaderModule result = VK_NULL_HANDLE;

	if (vkCreateShaderModule(graphics::internal::context.device, &shader_module,
							 nullptr, &result) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module: " << path << '\n';
		return VK_NULL_HANDLE;
	}

	return result;
}

// Создаёт буфер в памяти, доступной процессору, и оставляет её отображённой,
// чтобы данные можно было переписывать каждый кадр.
bool createMappedBuffer(VkDeviceSize size, VkBufferUsageFlags usage, MappedBuffer* const out) {
	auto& context = graphics::internal::context;

	const VkBufferCreateInfo buffer = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
				 VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo allocation_info = {};

	if (vmaCreateBuffer(context.allocator, &buffer, &allocation,
						&out->buffer, &out->allocation, &allocation_info) != VK_SUCCESS) {
		std::cerr << "Failed to create buffer\n";
		return false;
	}

	out->data = allocation_info.pMappedData;

	return true;
}

// Копирует данные в буфер и сбрасывает кеш, чтобы их увидела видеокарта.
void upload(const MappedBuffer& buffer, const void* const data, size_t size) {
	if (size == 0) {
		return;
	}

	std::memcpy(buffer.data, data, size);
	vmaFlushAllocation(graphics::internal::context.allocator, buffer.allocation, 0, size);
}

void destroyBuffer(const MappedBuffer& buffer) {
	vmaDestroyBuffer(graphics::internal::context.allocator, buffer.buffer, buffer.allocation);
}

// Записывает вектор и четвёртое число в массив из четырёх float.
void store(float* const out, Vec3 v, float w) {
	out[0] = v.x;
	out[1] = v.y;
	out[2] = v.z;
	out[3] = w;
}

// Нормализует направление; нулевой вектор заменяет направлением вниз.
Vec3 direction(Vec3 v) {
	const float length = std::sqrt(math::dot(v, v));
	return length > 0.0001f ? v * (1.0f / length) : Vec3{ 0.0f, -1.0f, 0.0f };
}

// Строит меш с плоскими нормалями: каждый треугольник получает свои три
// вершины, а нормаль считается векторным произведением двух его сторон.
std::vector<Vertex> buildFlatMesh(const Vec3* const positions, const uint16_t* const indices,
								  size_t index_count) {
	std::vector<Vertex> vertices;
	vertices.reserve(index_count);

	for (size_t i = 0; i < index_count; i += 3) {
		const Vec3 a = positions[indices[i]];
		const Vec3 b = positions[indices[i + 1]];
		const Vec3 c = positions[indices[i + 2]];

		const Vec3 n = math::normalize(math::cross(b - a, c - a));

		vertices.push_back({ { a.x, a.y, a.z }, { n.x, n.y, n.z } });
		vertices.push_back({ { b.x, b.y, b.z }, { n.x, n.y, n.z } });
		vertices.push_back({ { c.x, c.y, c.z }, { n.x, n.y, n.z } });
	}

	return vertices;
}

// Создаёт вершинный буфер модели и её uniform-буфер с матрицей и материалом.
bool createModel(Model& model, const Vec3* const positions, const uint16_t* const indices,
				 size_t index_count) {
	const std::vector<Vertex> vertices = buildFlatMesh(positions, indices, index_count);
	const size_t size = vertices.size() * sizeof(Vertex);

	if (!createMappedBuffer(size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &model.vertices)) {
		return false;
	}

	upload(model.vertices, vertices.data(), size);
	model.vertex_count = uint32_t(vertices.size());

	return createMappedBuffer(sizeof(ModelData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
							  &model.uniform);
}

// Доп. задание 1: создаёт два набора дескрипторов — для сцены и для модели —
// и связывает их с буферами.
bool createDescriptors() {
	auto& context = graphics::internal::context;

	// Набор сцены: uniform-буфер и два storage-буфера с источниками
	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	// Набор модели: один uniform-буфер
	const VkDescriptorSetLayoutBinding model_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo scene_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(scene_bindings) / sizeof(scene_bindings[0]),
		.pBindings = scene_bindings,
	};

	const VkDescriptorSetLayoutCreateInfo model_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &model_binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &scene_layout, nullptr,
									&vk_scene_set_layout) != VK_SUCCESS ||
		vkCreateDescriptorSetLayout(context.device, &model_layout, nullptr,
									&vk_model_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layouts\n";
		return false;
	}

	// Пул, из которого выделяются наборы: один для сцены и по одному на модель
	const VkDescriptorPoolSize pool_sizes[] = {
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uint32_t(1 + model_count) },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
	};

	const VkDescriptorPoolCreateInfo pool = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = uint32_t(1 + model_count),
		.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
		.pPoolSizes = pool_sizes,
	};

	if (vkCreateDescriptorPool(context.device, &pool, nullptr,
							   &vk_descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	VkDescriptorSetAllocateInfo allocate = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk_descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &vk_scene_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &allocate,
								 &vk_scene_descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate scene descriptor set\n";
		return false;
	}

	allocate.pSetLayouts = &vk_model_set_layout;

	for (Model& model : models) {
		if (vkAllocateDescriptorSets(context.device, &allocate,
									 &model.descriptor_set) != VK_SUCCESS) {
			std::cerr << "Failed to allocate model descriptor set\n";
			return false;
		}
	}

	// Связываем наборы с буферами. Буферы не пересоздаются,
	// поэтому это делается один раз.
	const VkDescriptorBufferInfo scene_buffers[] = {
		{ scene_uniform.buffer, 0, VK_WHOLE_SIZE },
		{ point_light_storage.buffer, 0, VK_WHOLE_SIZE },
		{ spot_light_storage.buffer, 0, VK_WHOLE_SIZE },
	};

	const VkDescriptorType scene_types[] = {
		VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	};

	VkWriteDescriptorSet writes[3] = {};

	for (uint32_t i = 0; i < 3; ++i) {
		writes[i] = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = vk_scene_descriptor_set,
			.dstBinding = i,
			.descriptorCount = 1,
			.descriptorType = scene_types[i],
			.pBufferInfo = &scene_buffers[i],
		};
	}

	vkUpdateDescriptorSets(context.device, 3, writes, 0, nullptr);

	for (const Model& model : models) {
		const VkDescriptorBufferInfo model_buffer = { model.uniform.buffer, 0, VK_WHOLE_SIZE };

		const VkWriteDescriptorSet write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = model.descriptor_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &model_buffer,
		};

		vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
	}

	return true;
}

// Создаёт графический конвейер: шейдеры, формат вершины, тест глубины,
// отсечение нелицевых граней и раскладку двух наборов дескрипторов.
bool createPipeline() {
	auto& context = graphics::internal::context;

	VkShaderModule vertex_shader = createShaderModule("shaders/lit.vert.spv");
	VkShaderModule fragment_shader = createShaderModule("shaders/lit.frag.spv");

	if (vertex_shader == VK_NULL_HANDLE || fragment_shader == VK_NULL_HANDLE) {
		vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		return false;
	}

	// Набор 0 — сцена, набор 1 — модель
	const VkDescriptorSetLayout set_layouts[] = {
		vk_scene_set_layout,
		vk_model_set_layout,
	};

	const VkPipelineLayoutCreateInfo pipeline_layout = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = sizeof(set_layouts) / sizeof(set_layouts[0]),
		.pSetLayouts = set_layouts,
	};

	if (vkCreatePipelineLayout(context.device, &pipeline_layout, nullptr,
							   &vk_pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader,
			.pName = "main",
		},
	};

	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription vertex_attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, position),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, normal),
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount =
			sizeof(vertex_attributes) / sizeof(vertex_attributes[0]),
		.pVertexAttributeDescriptions = vertex_attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	const VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};

	const VkPipelineColorBlendAttachmentState color_blend_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
						  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
		.pDynamicStates = dynamic_states,
	};

	const VkGraphicsPipelineCreateInfo pipeline = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = sizeof(stages) / sizeof(stages[0]),
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic_state,
		.layout = vk_pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
													  &pipeline, nullptr, &vk_pipeline);

	vkDestroyShaderModule(context.device, vertex_shader, nullptr);
	vkDestroyShaderModule(context.device, fragment_shader, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		return false;
	}

	return true;
}

// Доп. задание 2: список точечных источников с добавлением и удалением.
void drawPointLights() {
	// Своя область имён, чтобы элементы не совпадали по ID с прожекторами
	ImGui::PushID("point_lights");

	int remove = -1;

	for (size_t i = 0; i < point_lights.size(); ++i) {
		PointLight& light = point_lights[i];

		ImGui::PushID(int(i));

		if (ImGui::TreeNodeEx("##light", ImGuiTreeNodeFlags_DefaultOpen,
							  "Point light %d", int(i) + 1)) {
			ImGui::DragFloat3("Position", &light.position.x, 0.05f);
			ImGui::ColorEdit3("Color", &light.color.x);
			ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 50.0f);

			if (ImGui::Button("Remove")) {
				remove = int(i);
			}

			ImGui::TreePop();
		}

		ImGui::PopID();
	}

	if (remove >= 0) {
		point_lights.erase(point_lights.begin() + remove);
	}

	ImGui::BeginDisabled(point_lights.size() >= max_lights);

	if (ImGui::Button("Add point light")) {
		point_lights.push_back({ { 0.0f, 2.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, 5.0f });
	}

	ImGui::EndDisabled();
	ImGui::PopID();
}

// Доп. задание 3: список прожекторов с добавлением и удалением.
void drawSpotLights() {
	ImGui::PushID("spot_lights");

	int remove = -1;

	for (size_t i = 0; i < spot_lights.size(); ++i) {
		SpotLight& light = spot_lights[i];

		ImGui::PushID(int(i));

		if (ImGui::TreeNodeEx("##light", ImGuiTreeNodeFlags_DefaultOpen,
							  "Spot light %d", int(i) + 1)) {
			ImGui::DragFloat3("Position", &light.position.x, 0.05f);
			ImGui::DragFloat3("Direction", &light.direction.x, 0.01f, -1.0f, 1.0f);
			ImGui::ColorEdit3("Color", &light.color.x);
			ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 100.0f);
			ImGui::SliderFloat("Inner angle", &light.inner_angle, 1.0f, 89.0f, "%.0f deg");
			ImGui::SliderFloat("Outer angle", &light.outer_angle, 1.0f, 89.0f, "%.0f deg");

			// Внешний угол не может быть меньше внутреннего
			if (light.outer_angle < light.inner_angle) {
				light.outer_angle = light.inner_angle;
			}

			if (ImGui::Button("Remove")) {
				remove = int(i);
			}

			ImGui::TreePop();
		}

		ImGui::PopID();
	}

	if (remove >= 0) {
		spot_lights.erase(spot_lights.begin() + remove);
	}

	ImGui::BeginDisabled(spot_lights.size() >= max_lights);

	if (ImGui::Button("Add spot light")) {
		spot_lights.push_back({ { 0.0f, 4.0f, 0.0f }, { 0.0f, -1.0f, 0.0f },
								{ 1.0f, 1.0f, 1.0f }, 20.0f, 15.0f, 25.0f });
	}

	ImGui::EndDisabled();
	ImGui::PopID();
}

// Рисует окно интерфейса: камера, свет и материалы.
void drawInterface() {
	ImGui::SetNextWindowSize(ImVec2(420.0f, 680.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Lab 2 - Camera and Lighting");
	ImGui::PushItemWidth(-130.0f);

	ImGui::Text("%.1f FPS", double(ImGui::GetIO().Framerate));

	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Position##camera", &camera_position.x, 0.05f);
		ImGui::SliderFloat("Yaw", &camera_yaw, -180.0f, 180.0f, "%.0f deg");
		ImGui::SliderFloat("Pitch", &camera_pitch, -89.0f, 89.0f, "%.0f deg");
		ImGui::SliderFloat("Field of view", &camera_fov, 20.0f, 120.0f, "%.0f deg");
	}

	if (ImGui::CollapsingHeader("Ambient light", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Color##ambient", &ambient_color.x);
		ImGui::SliderFloat("Intensity##ambient", &ambient_intensity, 0.0f, 1.0f);
	}

	if (ImGui::CollapsingHeader("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Direction##sun", &sun_direction.x, 0.01f, -1.0f, 1.0f);
		ImGui::ColorEdit3("Color##sun", &sun_color.x);
		ImGui::SliderFloat("Intensity##sun", &sun_intensity, 0.0f, 5.0f);
	}

	if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		drawPointLights();
	}

	if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		drawSpotLights();
	}

	if (ImGui::CollapsingHeader("Materials", ImGuiTreeNodeFlags_DefaultOpen)) {
		for (Model& model : models) {
			ImGui::PushID(model.name);
			ImGui::SeparatorText(model.name);
			ImGui::ColorEdit3("Albedo", &model.albedo.x);
			ImGui::ColorEdit3("Specular color", &model.specular.x);
			ImGui::SliderFloat("Shininess", &model.shininess, 1.0f, 256.0f, "%.0f",
							   ImGuiSliderFlags_Logarithmic);
			ImGui::PopID();
		}
	}

	ImGui::PopItemWidth();
	ImGui::End();
}

// Переписывает все буферы сцены и моделей актуальными данными из интерфейса.
void uploadFrameData() {
	auto& context = graphics::internal::context;

	const float width = float(context.swapchain_extent.width);
	const float height = float(context.swapchain_extent.height);
	const float aspect = height > 0.0f ? width / height : 1.0f;

	// Камера: направление взгляда из двух углов
	const float yaw = math::radians(camera_yaw);
	const float pitch = math::radians(camera_pitch);

	const Vec3 forward = {
		std::cos(pitch) * std::sin(yaw),
		std::sin(pitch),
		-std::cos(pitch) * std::cos(yaw),
	};

	const Mat4 view = math::lookAt(camera_position, camera_position + forward,
								   { 0.0f, 1.0f, 0.0f });
	const Mat4 projection = math::perspective(math::radians(camera_fov), aspect,
											  0.1f, 100.0f);

	SceneData scene = {};
	scene.view_projection = projection * view;
	store(scene.camera_position, camera_position, 1.0f);
	store(scene.ambient, ambient_color * ambient_intensity, 0.0f);
	store(scene.light_direction, direction(sun_direction), 0.0f);
	store(scene.light_color, sun_color, sun_intensity);
	scene.point_light_count = uint32_t(point_lights.size());
	scene.spot_light_count = uint32_t(spot_lights.size());

	upload(scene_uniform, &scene, sizeof(scene));

	GpuPointLight gpu_points[max_lights] = {};

	for (size_t i = 0; i < point_lights.size(); ++i) {
		const PointLight& light = point_lights[i];
		store(gpu_points[i].position, light.position, 1.0f);
		store(gpu_points[i].color, light.color, light.intensity);
	}

	upload(point_light_storage, gpu_points, point_lights.size() * sizeof(GpuPointLight));

	GpuSpotLight gpu_spots[max_lights] = {};

	for (size_t i = 0; i < spot_lights.size(); ++i) {
		const SpotLight& light = spot_lights[i];
		store(gpu_spots[i].position, light.position, 1.0f);
		store(gpu_spots[i].direction, direction(light.direction), 0.0f);
		store(gpu_spots[i].color, light.color, light.intensity);

		// В шейдер уходят косинусы: сравнивать косинусы дешевле, чем считать углы
		gpu_spots[i].cutoff[0] = std::cos(math::radians(light.inner_angle));
		gpu_spots[i].cutoff[1] = std::cos(math::radians(light.outer_angle));
	}

	upload(spot_light_storage, gpu_spots, spot_lights.size() * sizeof(GpuSpotLight));

	for (const Model& model : models) {
		ModelData data = {};
		data.transform = math::translation(model.position);
		store(data.albedo, model.albedo, 1.0f);
		store(data.specular, model.specular, model.shininess);

		upload(model.uniform, &data, sizeof(data));
	}
}

} // namespace

// Вызывается один раз при запуске: буферы, наборы дескрипторов и конвейер.
bool initialize() {
	if (!createMappedBuffer(sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
							&scene_uniform) ||
		!createMappedBuffer(sizeof(GpuPointLight) * max_lights,
							VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &point_light_storage) ||
		!createMappedBuffer(sizeof(GpuSpotLight) * max_lights,
							VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &spot_light_storage)) {
		return false;
	}

	if (!createModel(models[0], octahedron_positions, octahedron_indices,
					 sizeof(octahedron_indices) / sizeof(octahedron_indices[0])) ||
		!createModel(models[1], floor_positions, floor_indices,
					 sizeof(floor_indices) / sizeof(floor_indices[0]))) {
		return false;
	}

	if (!createDescriptors()) {
		return false;
	}

	return createPipeline();
}

// Вызывается один раз при выходе: уничтожает всё, созданное в initialize.
void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	for (const Model& model : models) {
		destroyBuffer(model.vertices);
		destroyBuffer(model.uniform);
	}

	destroyBuffer(spot_light_storage);
	destroyBuffer(point_light_storage);
	destroyBuffer(scene_uniform);

	// Наборы дескрипторов освобождаются вместе с пулом
	vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_model_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_scene_set_layout, nullptr);

	vkDestroyPipeline(context.device, vk_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);
}

// Вызывается каждый кадр: рисует интерфейс.
void update([[maybe_unused]] double time) {
	drawInterface();
}

// Вызывается каждый кадр: обновляет буферы и записывает команды отрисовки.
void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

	if (vk_pipeline == VK_NULL_HANDLE) {
		return;
	}

	// Буферы переписываются здесь, а не в update: к этому моменту стартовый код
	// уже дождался окончания прошлого кадра, и видеокарта их не читает
	uploadFrameData();

	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = float(context.swapchain_extent.width),
		.height = float(context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};

	const VkRect2D scissor = {
		.offset = { 0, 0 },
		.extent = context.swapchain_extent,
	};

	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

	// Набор сцены привязывается один раз на кадр
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
							vk_pipeline_layout, 0, 1, &vk_scene_descriptor_set, 0, nullptr);

	// Набор модели — перед каждым объектом
	for (const Model& model : models) {
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
								vk_pipeline_layout, 1, 1, &model.descriptor_set, 0, nullptr);

		const VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &model.vertices.buffer, &offset);

		vkCmdDraw(fd.command_buffer, model.vertex_count, 1, 0, 0);
	}
}

} // namespace application
