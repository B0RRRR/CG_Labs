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

struct Vertex {
	float position[3];
	float color[3];
};

// Данные, которые передаются в шейдер напрямую
struct PushConstants {
	Mat4 mvp;
	float tint[4];
};


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

constexpr uint32_t octahedron_index_count =
	sizeof(octahedron_indices) / sizeof(octahedron_indices[0]);


VkPipelineLayout vk_pipeline_layout;
VkPipeline vk_pipeline;

VkBuffer vk_vertex_buffer;
VmaAllocation vma_vertex_allocation;

VkBuffer vk_index_buffer;
VmaAllocation vma_index_allocation;

// Состояние, которым управляет интерфейс
bool use_perspective = true;
float field_of_view = 60.0f;
float ortho_height = 4.0f;
float camera_distance = 4.0f;

Vec3 position = { 0.0f, 0.0f, 0.0f };
Vec3 rotation = { 0.0f, 0.0f, 0.0f };
Vec3 scale = { 1.0f, 1.0f, 1.0f };

bool animation_playing = false;
float animation_speed = 1.0f;
float animation_radius = 1.5f;
float animation_height = 0.7f;
float animation_spin = 90.0f;
float animation_time = 0.0f;

float tint_color[3] = { 1.0f, 1.0f, 1.0f };
bool use_vertex_colors = true;

Mat4 model_matrix = math::identity();

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

bool createBuffer(const void* const data, VkDeviceSize size, VkBufferUsageFlags usage,
				  VkBuffer* const out_buffer, VmaAllocation* const out_allocation) {
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
						out_buffer, out_allocation, &allocation_info) != VK_SUCCESS) {
		std::cerr << "Failed to create buffer\n";
		return false;
	}

	std::memcpy(allocation_info.pMappedData, data, size_t(size));

	return vmaFlushAllocation(context.allocator, *out_allocation, 0, size) == VK_SUCCESS;
}

bool createPipeline() {
	auto& context = graphics::internal::context;

	VkShaderModule vertex_shader = createShaderModule("shaders/octahedron.vert.spv");
	VkShaderModule fragment_shader = createShaderModule("shaders/octahedron.frag.spv");

	if (vertex_shader == VK_NULL_HANDLE || fragment_shader == VK_NULL_HANDLE) {
		vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		return false;
	}

	const VkPushConstantRange push_constants = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		.offset = 0,
		.size = sizeof(PushConstants),
	};

	const VkPipelineLayoutCreateInfo pipeline_layout = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_constants,
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
			.offset = offsetof(Vertex, color),
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

bool createMesh() {
	constexpr size_t vertex_count =
		sizeof(octahedron_positions) / sizeof(octahedron_positions[0]);

	Vertex vertices[vertex_count] = {};

	for (size_t i = 0; i < vertex_count; ++i) {
		const Vec3 p = octahedron_positions[i];

		vertices[i].position[0] = p.x;
		vertices[i].position[1] = p.y;
		vertices[i].position[2] = p.z;

		vertices[i].color[0] = p.x * 0.5f + 0.5f;
		vertices[i].color[1] = p.y * 0.5f + 0.5f;
		vertices[i].color[2] = p.z * 0.5f + 0.5f;
	}

	if (!createBuffer(vertices, sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
					  &vk_vertex_buffer, &vma_vertex_allocation)) {
		return false;
	}

	return createBuffer(octahedron_indices, sizeof(octahedron_indices),
						VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
						&vk_index_buffer, &vma_index_allocation);
}

Vec3 animationOffset() {
	if (animation_radius == 0.0f && animation_height == 0.0f) {
		return { 0.0f, 0.0f, 0.0f };
	}

	const float t = animation_time;

	return {
		animation_radius * std::cos(t),
		animation_height * std::sin(t * 2.0f),
		animation_radius * std::sin(t),
	};
}

void drawInterface() {
	ImGui::Begin("Lab 1 - Octahedron");

	ImGui::Text("%.1f FPS (%.2f ms/frame)",
				double(ImGui::GetIO().Framerate),
				double(1000.0f / ImGui::GetIO().Framerate));

	if (ImGui::CollapsingHeader("Projection", ImGuiTreeNodeFlags_DefaultOpen)) {
		int projection = use_perspective ? 0 : 1;

		ImGui::RadioButton("Perspective", &projection, 0);
		ImGui::SameLine();
		ImGui::RadioButton("Orthographic", &projection, 1);

		use_perspective = projection == 0;

		if (use_perspective) {
			ImGui::SliderFloat("Field of view", &field_of_view, 20.0f, 120.0f, "%.0f deg");
		} else {
			ImGui::SliderFloat("View height", &ortho_height, 1.0f, 20.0f);
		}

		ImGui::SliderFloat("Camera distance", &camera_distance, 2.0f, 20.0f);
	}

	if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Position", &position.x, 0.01f, -5.0f, 5.0f);
		ImGui::DragFloat3("Rotation", &rotation.x, 1.0f, -360.0f, 360.0f, "%.0f deg");
		ImGui::DragFloat3("Scale", &scale.x, 0.01f, 0.05f, 5.0f);

		if (ImGui::Button("Reset transform")) {
			position = { 0.0f, 0.0f, 0.0f };
			rotation = { 0.0f, 0.0f, 0.0f };
			scale = { 1.0f, 1.0f, 1.0f };
		}
	}

	if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::Button(animation_playing ? "Pause" : "Play")) {
			animation_playing = !animation_playing;
		}

		ImGui::SameLine();

		if (ImGui::Button("Restart")) {
			animation_time = 0.0f;
		}

		ImGui::SliderFloat("Speed", &animation_speed, 0.0f, 5.0f);
		ImGui::SliderFloat("Radius", &animation_radius, 0.0f, 4.0f);
		ImGui::SliderFloat("Height", &animation_height, 0.0f, 3.0f);
		ImGui::SliderFloat("Spin", &animation_spin, -360.0f, 360.0f, "%.0f deg/s");
	}

	if (ImGui::CollapsingHeader("Color", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Tint", tint_color);
		ImGui::Checkbox("Procedural vertex colors", &use_vertex_colors);
		ImGui::TextWrapped("Vertex colors come from vertex positions in local space; "
						   "the tint above is multiplied by them.");
	}

	ImGui::End();
}

} // namespace

bool initialize() {
	if (!createPipeline()) {
		return false;
	}

	return createMesh();
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	vmaDestroyBuffer(context.allocator, vk_index_buffer, vma_index_allocation);
	vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vma_vertex_allocation);

	vkDestroyPipeline(context.device, vk_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);
}

void update(double time) {
	static double previous_time = time;

	const float delta_time = float(time - previous_time);
	previous_time = time;

	if (animation_playing) {
		animation_time += delta_time * animation_speed;
	}

	drawInterface();

	// Матрица модели
	const Vec3 animated_position = position + animationOffset();

	const Vec3 animated_rotation = {
		math::radians(rotation.x),
		math::radians(rotation.y + animation_time * animation_spin),
		math::radians(rotation.z),
	};

	model_matrix = math::translation(animated_position) *
				   math::rotation(animated_rotation) *
				   math::scaling(scale);
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

	if (vk_pipeline == VK_NULL_HANDLE) {
		return;
	}

	const float width = float(context.swapchain_extent.width);
	const float height = float(context.swapchain_extent.height);
	const float aspect = height > 0.0f ? width / height : 1.0f;

	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = width,
		.height = height,
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

	// Доп. задание 1
	const Mat4 projection =
		use_perspective
			? math::perspective(math::radians(field_of_view), aspect, 0.1f, 100.0f)
			: math::orthographic(ortho_height, aspect, 0.1f, 100.0f);

	const Mat4 view = math::lookAt({ 0.0f, 0.0f, camera_distance },
								   { 0.0f, 0.0f, 0.0f },
								   { 0.0f, 1.0f, 0.0f });

	const PushConstants push_constants = {
		.mvp = projection * view * model_matrix,
		.tint = {
			tint_color[0],
			tint_color[1],
			tint_color[2],
			use_vertex_colors ? 1.0f : 0.0f,
		},
	};

	vkCmdPushConstants(fd.command_buffer, vk_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
					   0, sizeof(push_constants), &push_constants);

	const VkDeviceSize offset = 0;

	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_vertex_buffer, &offset);
	vkCmdBindIndexBuffer(fd.command_buffer, vk_index_buffer, 0, VK_INDEX_TYPE_UINT16);

	vkCmdDrawIndexed(fd.command_buffer, octahedron_index_count, 1, 0, 0, 0);
}

}