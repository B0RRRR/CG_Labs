#include "application.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>

// Загрузка и декодирование изображений
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#include <stb_image.h>

#include "math.hpp"

namespace application {

namespace {

using math::Mat4;
using math::Vec3;

// Сколько источников каждого типа помещается в storage-буфер
constexpr uint32_t max_lights = 64;

// Разрешение квадратной карты теней (в текселях)
constexpr uint32_t shadow_map_size = 2048;

struct Vertex {
	float position[3];
	float normal[3];
	float uv[2];
};

// Данные для видеокарты. Раскладка полей повторяет шейдеры

// Набор 0, привязка 0: общие данные сцены (uniform-буфер)
struct SceneData {
	Mat4 view_projection;
	Mat4 light_view_projection;
	float camera_position[4];
	float ambient[4];
	float light_direction[4];
	float light_color[4];
	uint32_t point_light_count;
	uint32_t spot_light_count;
	float shadow_bias;
	uint32_t shadow_pcf;
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

// Набор 1, привязка 0: элемент массива свойств моделей (динамический uniform-буфер)
struct ModelData {
	Mat4 transform;
	float specular[4];
};

// Push-константы фрагментного шейдера (анимированное сэмплирование из 3-й работы)
struct EffectConstants {
	uint32_t enabled;
	float time;
	float speed;
	float distortion;
};

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

struct MappedBuffer {
	VkBuffer buffer;
	VmaAllocation allocation;
	void* data;
};

// Буфер в памяти видеокарты (VRAM), процессору недоступен
struct GpuBuffer {
	VkBuffer buffer;
	VmaAllocation allocation;
};

// Геометрия: вершины и индексы в VRAM. Одну сетку могут рисовать несколько моделей
struct Mesh {
	GpuBuffer vertices;
	GpuBuffer indices;
	uint32_t index_count;
};

struct Texture {
	const char* name;
	const char* path;

	VkImage image;
	VmaAllocation allocation;
	VkImageView view;
	uint32_t width;
	uint32_t height;
	uint32_t mip_levels;

	VkDescriptorSet descriptor_set;
};

enum MeshId : uint32_t {
	mesh_octahedron,
	mesh_floor,
	mesh_wall,
	mesh_count,
};

enum TextureId : int {
	texture_bricks,
	texture_wood,
	texture_marble,
	texture_lava,
	texture_checker,
	texture_tiles,
};

struct Model {
	const char* name;
	MeshId mesh;
	int texture;

	Vec3 position;
	float rotation;   // поворот вокруг вертикальной оси, в градусах
	float spin_speed; // скорость вращения, градусов в секунду
	float scale;

	Vec3 specular;
	float shininess;

	bool animated;
	float effect_speed;
	float effect_distortion;
};

// Промежуточные данные сетки на процессоре
struct MeshData {
	std::vector<Vertex> vertices;
	std::vector<uint16_t> indices;
};

constexpr Vec3 octahedron_positions[] = {
	{  1.0f,  0.0f,  0.0f },
	{ -1.0f,  0.0f,  0.0f },
	{  0.0f,  1.0f,  0.0f },
	{  0.0f, -1.0f,  0.0f },
	{  0.0f,  0.0f,  1.0f },
	{  0.0f,  0.0f, -1.0f },
};

// Треугольники перечислены против часовой стрелки при взгляде снаружи
constexpr uint16_t octahedron_indices[] = {
	2, 4, 0,  2, 1, 4,  2, 5, 1,  2, 0, 5,
	3, 0, 4,  3, 4, 1,  3, 1, 5,  3, 5, 0,
};

constexpr float room_size = 6.0f;   // половина ширины комнаты
constexpr float wall_height = 4.5f;

// Объекты Vulkan

VkDescriptorSetLayout vk_scene_set_layout;
VkDescriptorSetLayout vk_material_set_layout;
VkDescriptorSetLayout vk_shadow_scene_set_layout;
VkDescriptorSetLayout vk_shadow_model_set_layout;
VkDescriptorPool vk_descriptor_pool;
VkDescriptorSet vk_scene_descriptor_set;
VkDescriptorSet vk_shadow_scene_descriptor_set;
VkDescriptorSet vk_shadow_model_descriptor_set;

VkPipelineLayout vk_pipeline_layout;
VkPipeline vk_pipeline;

// Конвейер и ресурсы карты теней (Dynamic Rendering, только вершинный шейдер)
VkPipelineLayout vk_shadow_pipeline_layout;
VkPipeline vk_shadow_pipeline;

VkFormat vk_shadow_format;
bool shadow_linear_filter = false;
VkImage vk_shadow_image;
VmaAllocation vma_shadow_allocation;
VkImageView vk_shadow_image_view;
VkSampler vk_shadow_sampler;

VkSampler vk_sampler;

MappedBuffer scene_uniform;
MappedBuffer point_light_storage;
MappedBuffer spot_light_storage;

// Один динамический uniform-буфер на все модели и шаг между его элементами
MappedBuffer model_uniform;
VkDeviceSize model_stride;

// Пул и буфер команд для копирования данных в VRAM при запуске
VkCommandPool vk_upload_pool;
VkCommandBuffer vk_upload_commands;
std::vector<MappedBuffer> staging_buffers;

Mesh meshes[mesh_count];

Texture textures[] = {
	{ .name = "Bricks", .path = "labs/lab3/textures/bricks.png" },
	{ .name = "Wood planks", .path = "labs/lab3/textures/wood.png" },
	{ .name = "Marble", .path = "labs/lab3/textures/marble.png" },
	{ .name = "Lava", .path = "labs/lab3/textures/lava.png" },
	{ .name = "UV checker", .path = "labs/lab3/textures/checker.png" },
	{ .name = "Stone tiles", .path = "labs/lab3/textures/tiles.png" },
};

constexpr size_t texture_count = sizeof(textures) / sizeof(textures[0]);

// Состояние, которым управляет интерфейс
Vec3 camera_position = { 0.0f, 1.5f, 5.8f };
float camera_yaw = 0.0f;
float camera_pitch = -15.0f;
float camera_fov = 70.0f;

Vec3 ambient_color = { 1.0f, 1.0f, 1.0f };
float ambient_intensity = 0.15f;

Vec3 sun_direction = { -0.45f, -1.0f, -0.55f };
Vec3 sun_color = { 1.0f, 0.96f, 0.9f };
float sun_intensity = 1.2f;

// Параметры карты теней, управляемые через интерфейс
float shadow_ortho_size = 16.0f;  // сторона квадрата ортографической проекции
float shadow_depth_bias = 0.0015f; // постоянный сдвиг глубины (в шейдере)
float shadow_slope_bias = 1.5f;    // аппаратный наклонный сдвиг (vkCmdSetDepthBias)
bool shadow_pcf = true;            // сглаживание границ тени (3x3 PCF)

std::vector<PointLight> point_lights = {
	{ { 2.0f, 1.0f, 2.0f }, { 1.0f, 0.6f, 0.3f }, 2.5f },
};

std::vector<SpotLight> spot_lights = {
	{ { -2.5f, 2.8f, 2.0f }, { 0.3f, -1.0f, -0.4f }, { 0.4f, 0.7f, 1.0f }, 6.0f, 15.0f, 28.0f },
};

Model models[] = {
	{
		.name = "Octahedron",
		.mesh = mesh_octahedron,
		.texture = texture_lava,
		.position = { 0.0f, 0.0f, 0.0f },
		.rotation = 0.0f,
		.spin_speed = 15.0f,
		.scale = 1.0f,
		.specular = { 1.0f, 1.0f, 1.0f },
		.shininess = 64.0f,
		.animated = true,
		.effect_speed = 1.0f,
		.effect_distortion = 0.02f,
	},
	{
		.name = "Left octahedron",
		.mesh = mesh_octahedron,
		.texture = texture_checker,
		.position = { -2.6f, -0.8f, 0.5f },
		.rotation = 0.0f,
		.spin_speed = 20.0f,
		.scale = 0.7f,
		.specular = { 0.6f, 0.6f, 0.6f },
		.shininess = 32.0f,
		.animated = false,
		.effect_speed = 1.0f,
		.effect_distortion = 0.02f,
	},
	{
		.name = "Right octahedron",
		.mesh = mesh_octahedron,
		.texture = texture_marble,
		.position = { 2.6f, -0.8f, 0.5f },
		.rotation = 0.0f,
		.spin_speed = -20.0f,
		.scale = 0.7f,
		.specular = { 1.0f, 1.0f, 1.0f },
		.shininess = 128.0f,
		.animated = false,
		.effect_speed = 1.0f,
		.effect_distortion = 0.02f,
	},
	{
		.name = "Floor",
		.mesh = mesh_floor,
		.texture = texture_wood,
		.position = { 0.0f, -1.5f, 0.0f },
		.rotation = 0.0f,
		.spin_speed = 0.0f,
		.scale = 1.0f,
		.specular = { 0.3f, 0.3f, 0.3f },
		.shininess = 24.0f,
		.animated = false,
		.effect_speed = 1.0f,
		.effect_distortion = 0.02f,
	},
	{
		.name = "Back wall",
		.mesh = mesh_wall,
		.texture = texture_bricks,
		.position = { 0.0f, -1.5f, -room_size },
		.rotation = 0.0f,
		.spin_speed = 0.0f,
		.scale = 1.0f,
		.specular = { 0.1f, 0.1f, 0.1f },
		.shininess = 8.0f,
		.animated = false,
		.effect_speed = 1.0f,
		.effect_distortion = 0.02f,
	},
	{
		.name = "Left wall",
		.mesh = mesh_wall,
		.texture = texture_bricks,
		.position = { -room_size, -1.5f, 0.0f },
		.rotation = 90.0f,
		.spin_speed = 0.0f,
		.scale = 1.0f,
		.specular = { 0.1f, 0.1f, 0.1f },
		.shininess = 8.0f,
		.animated = false,
		.effect_speed = 1.0f,
		.effect_distortion = 0.02f,
	},
	{
		.name = "Right wall",
		.mesh = mesh_wall,
		.texture = texture_bricks,
		.position = { room_size, -1.5f, 0.0f },
		.rotation = -90.0f,
		.spin_speed = 0.0f,
		.scale = 1.0f,
		.specular = { 0.1f, 0.1f, 0.1f },
		.shininess = 8.0f,
		.animated = false,
		.effect_speed = 1.0f,
		.effect_distortion = 0.02f,
	},
};

constexpr size_t model_count = sizeof(models) / sizeof(models[0]);

float current_time = 0.0f;

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

// Создаёт буфер в памяти, доступной процессору, и оставляет её отображённой.
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

void destroyBuffer(const GpuBuffer& buffer) {
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

// Копирование в VRAM через staging-буферы

bool beginUpload() {
	auto& context = graphics::internal::context;

	const VkCommandPoolCreateInfo pool = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
		.queueFamilyIndex = context.graphics_queue_index,
	};

	if (vkCreateCommandPool(context.device, &pool, nullptr, &vk_upload_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create upload command pool\n";
		return false;
	}

	const VkCommandBufferAllocateInfo allocate = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = vk_upload_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};

	if (vkAllocateCommandBuffers(context.device, &allocate, &vk_upload_commands) != VK_SUCCESS) {
		std::cerr << "Failed to allocate upload command buffer\n";
		return false;
	}

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	return vkBeginCommandBuffer(vk_upload_commands, &begin) == VK_SUCCESS;
}

const MappedBuffer* stage(const void* const data, size_t size) {
	MappedBuffer staging = {};

	if (!createMappedBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &staging)) {
		return nullptr;
	}

	upload(staging, data, size);
	staging_buffers.push_back(staging);

	return &staging_buffers.back();
}

bool createDeviceBuffer(const void* const data, size_t size, VkBufferUsageFlags usage,
						GpuBuffer* const out) {
	auto& context = graphics::internal::context;

	const VkBufferCreateInfo buffer = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};

	if (vmaCreateBuffer(context.allocator, &buffer, &allocation,
						&out->buffer, &out->allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create device buffer\n";
		return false;
	}

	const MappedBuffer* const staging = stage(data, size);
	if (staging == nullptr) {
		return false;
	}

	const VkBufferCopy region = { .srcOffset = 0, .dstOffset = 0, .size = size };
	vkCmdCopyBuffer(vk_upload_commands, staging->buffer, out->buffer, 1, &region);

	return true;
}

bool endUpload() {
	auto& context = graphics::internal::context;

	const VkMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT,
	};

	vkCmdPipelineBarrier(vk_upload_commands, VK_PIPELINE_STAGE_TRANSFER_BIT,
						 VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 1, &barrier, 0, nullptr, 0,
						 nullptr);

	bool ok = vkEndCommandBuffer(vk_upload_commands) == VK_SUCCESS;

	if (ok) {
		const VkSubmitInfo submit = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers = &vk_upload_commands,
		};

		ok = vkQueueSubmit(context.graphics_queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS &&
			 vkQueueWaitIdle(context.graphics_queue) == VK_SUCCESS;
	}

	if (!ok) {
		std::cerr << "Failed to upload data to the GPU\n";
	}

	for (const MappedBuffer& staging : staging_buffers) {
		destroyBuffer(staging);
	}

	staging_buffers.clear();

	vkDestroyCommandPool(context.device, vk_upload_pool, nullptr);
	vk_upload_pool = VK_NULL_HANDLE;

	return ok;
}

// Сетки

void addVertex(MeshData& mesh, const Vertex& vertex) {
	for (size_t i = 0; i < mesh.vertices.size(); ++i) {
		if (std::memcmp(&mesh.vertices[i], &vertex, sizeof(Vertex)) == 0) {
			mesh.indices.push_back(uint16_t(i));
			return;
		}
	}

	mesh.indices.push_back(uint16_t(mesh.vertices.size()));
	mesh.vertices.push_back(vertex);
}

float sign(float v) {
	return v < 0.0f ? -1.0f : 1.0f;
}

void octahedralUv(Vec3 p, float sx, float sz, bool upper, float* const uv) {
	float u = p.x;
	float v = p.z;

	if (!upper) {
		u = (1.0f - std::abs(p.z)) * sx;
		v = (1.0f - std::abs(p.x)) * sz;
	}

	uv[0] = 0.5f + 0.5f * u;
	uv[1] = 0.5f + 0.5f * v;
}

MeshData buildOctahedron() {
	MeshData mesh;

	for (size_t i = 0; i < sizeof(octahedron_indices) / sizeof(octahedron_indices[0]); i += 3) {
		const Vec3 corners[3] = {
			octahedron_positions[octahedron_indices[i]],
			octahedron_positions[octahedron_indices[i + 1]],
			octahedron_positions[octahedron_indices[i + 2]],
		};

		const Vec3 n = math::normalize(
			math::cross(corners[1] - corners[0], corners[2] - corners[0]));

		const float sx = sign(n.x);
		const float sz = sign(n.z);
		const bool upper = n.y > 0.0f;

		for (const Vec3& p : corners) {
			Vertex vertex = { { p.x, p.y, p.z }, { n.x, n.y, n.z }, {} };
			octahedralUv(p, sx, sz, upper, vertex.uv);
			addVertex(mesh, vertex);
		}
	}

	return mesh;
}

MeshData buildQuad(Vec3 origin, Vec3 edge_u, Vec3 edge_v, float tiles_u, float tiles_v) {
	const Vec3 n = math::normalize(math::cross(edge_u, edge_v));

	const float corners[4][2] = { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } };
	const int order[6] = { 0, 1, 2, 0, 2, 3 };

	MeshData mesh;

	for (const int c : order) {
		const float a = corners[c][0];
		const float b = corners[c][1];
		const Vec3 p = origin + edge_u * a + edge_v * b;

		addVertex(mesh, { { p.x, p.y, p.z }, { n.x, n.y, n.z },
						  { a * tiles_u, (1.0f - b) * tiles_v } });
	}

	return mesh;
}

bool createMesh(const MeshData& data, Mesh* const out) {
	out->index_count = uint32_t(data.indices.size());

	return createDeviceBuffer(data.vertices.data(), data.vertices.size() * sizeof(Vertex),
							  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &out->vertices) &&
		   createDeviceBuffer(data.indices.data(), data.indices.size() * sizeof(uint16_t),
							  VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &out->indices);
}

bool createMeshes() {
	const float size = room_size;

	const MeshData floor = buildQuad({ -size, 0.0f, -size }, { 0.0f, 0.0f, 2.0f * size },
									 { 2.0f * size, 0.0f, 0.0f }, 4.0f, 4.0f);

	const MeshData wall = buildQuad({ -size, 0.0f, 0.0f }, { 2.0f * size, 0.0f, 0.0f },
									{ 0.0f, wall_height, 0.0f }, 4.0f, wall_height / 3.0f);

	return createMesh(buildOctahedron(), &meshes[mesh_octahedron]) &&
		   createMesh(floor, &meshes[mesh_floor]) &&
		   createMesh(wall, &meshes[mesh_wall]);
}

// Текстуры

void imageBarrier(VkImage image, uint32_t base_mip, uint32_t mip_count,
				  VkImageLayout old_layout, VkImageLayout new_layout,
				  VkAccessFlags src_access, VkAccessFlags dst_access,
				  VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
	const VkImageMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = src_access,
		.dstAccessMask = dst_access,
		.oldLayout = old_layout,
		.newLayout = new_layout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = base_mip,
			.levelCount = mip_count,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};

	vkCmdPipelineBarrier(vk_upload_commands, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr,
						 1, &barrier);
}

bool canGenerateMipmaps(VkFormat format) {
	VkFormatProperties properties = {};
	vkGetPhysicalDeviceFormatProperties(graphics::internal::context.physical_device, format,
										&properties);

	const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_BLIT_SRC_BIT |
										  VK_FORMAT_FEATURE_BLIT_DST_BIT |
										  VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;

	return (properties.optimalTilingFeatures & required) == required;
}

void generateMipmaps(const Texture& texture) {
	int32_t width = int32_t(texture.width);
	int32_t height = int32_t(texture.height);

	for (uint32_t level = 1; level < texture.mip_levels; ++level) {
		imageBarrier(texture.image, level - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
					 VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
					 VK_PIPELINE_STAGE_TRANSFER_BIT);

		const int32_t next_width = std::max(width / 2, 1);
		const int32_t next_height = std::max(height / 2, 1);

		const VkImageBlit blit = {
			.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1 },
			.srcOffsets = { { 0, 0, 0 }, { width, height, 1 } },
			.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1 },
			.dstOffsets = { { 0, 0, 0 }, { next_width, next_height, 1 } },
		};

		vkCmdBlitImage(vk_upload_commands, texture.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
					   texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
					   VK_FILTER_LINEAR);

		imageBarrier(texture.image, level - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
					 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
					 VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
					 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

		width = next_width;
		height = next_height;
	}

	imageBarrier(texture.image, texture.mip_levels - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
				 VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
				 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}

bool loadTexture(Texture& texture, bool mipmaps) {
	auto& context = graphics::internal::context;

	int width = 0, height = 0, channels = 0;
	stbi_uc* const pixels = stbi_load(texture.path, &width, &height, &channels, STBI_rgb_alpha);

	if (pixels == nullptr) {
		std::cerr << "Failed to load image " << texture.path << ": " << stbi_failure_reason()
				  << '\n';
		return false;
	}

	texture.width = uint32_t(width);
	texture.height = uint32_t(height);

	texture.mip_levels = 1;

	if (mipmaps) {
		for (uint32_t side = std::max(texture.width, texture.height); side > 1; side /= 2) {
			++texture.mip_levels;
		}
	}

	const VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;

	const VkImageCreateInfo image = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = format,
		.extent = { texture.width, texture.height, 1 },
		.mipLevels = texture.mip_levels,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
				 VK_IMAGE_USAGE_SAMPLED_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};

	const VmaAllocationCreateInfo allocation = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};

	if (vmaCreateImage(context.allocator, &image, &allocation, &texture.image,
					   &texture.allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create image for " << texture.path << '\n';
		stbi_image_free(pixels);
		return false;
	}

	const MappedBuffer* const staging = stage(pixels, size_t(width) * size_t(height) * 4);
	stbi_image_free(pixels);

	if (staging == nullptr) {
		return false;
	}

	imageBarrier(texture.image, 0, texture.mip_levels, VK_IMAGE_LAYOUT_UNDEFINED,
				 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
				 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

	const VkBufferImageCopy region = {
		.bufferOffset = 0,
		.bufferRowLength = 0,
		.bufferImageHeight = 0,
		.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
		.imageOffset = { 0, 0, 0 },
		.imageExtent = { texture.width, texture.height, 1 },
	};

	vkCmdCopyBufferToImage(vk_upload_commands, staging->buffer, texture.image,
						   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	generateMipmaps(texture);

	const VkImageViewCreateInfo view = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = texture.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = texture.mip_levels,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};

	if (vkCreateImageView(context.device, &view, nullptr, &texture.view) != VK_SUCCESS) {
		std::cerr << "Failed to create image view for " << texture.path << '\n';
		return false;
	}

	return true;
}

// Карта теней и сэмплеры

// Выбирает формат глубины без трафарета, поддерживающий рендеринг и выборку из шейдера.
VkFormat selectShadowDepthFormat(VkPhysicalDevice physical_device, bool* const linear_filter) {
	const VkFormat candidates[] = {
		VK_FORMAT_D32_SFLOAT,
		VK_FORMAT_D16_UNORM,
	};

	const VkFormatFeatureFlags required =
		VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;

	for (const VkFormat format : candidates) {
		VkFormatProperties properties = {};
		vkGetPhysicalDeviceFormatProperties(physical_device, format, &properties);

		if ((properties.optimalTilingFeatures &
			 (required | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) ==
			(required | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
			*linear_filter = true;
			return format;
		}
	}

	for (const VkFormat format : candidates) {
		VkFormatProperties properties = {};
		vkGetPhysicalDeviceFormatProperties(physical_device, format, &properties);

		if ((properties.optimalTilingFeatures & required) == required) {
			*linear_filter = false;
			return format;
		}
	}

	return VK_FORMAT_UNDEFINED;
}

// Создаёт текстуру глубины и её view для внеэкранной отрисовки карты теней.
bool createShadowMap() {
	auto& context = graphics::internal::context;

	vk_shadow_format = selectShadowDepthFormat(context.physical_device, &shadow_linear_filter);
	if (vk_shadow_format == VK_FORMAT_UNDEFINED) {
		std::cerr << "Failed to find supported depth format for shadow map\n";
		return false;
	}

	const VkImageCreateInfo image = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = vk_shadow_format,
		.extent = { shadow_map_size, shadow_map_size, 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};

	const VmaAllocationCreateInfo allocation = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};

	if (vmaCreateImage(context.allocator, &image, &allocation, &vk_shadow_image,
					   &vma_shadow_allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create shadow map image\n";
		return false;
	}

	const VkImageViewCreateInfo view = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = vk_shadow_image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = vk_shadow_format,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};

	if (vkCreateImageView(context.device, &view, nullptr, &vk_shadow_image_view) != VK_SUCCESS) {
		std::cerr << "Failed to create shadow map image view\n";
		return false;
	}

	return true;
}

// Создаёт обычный сэмплер для текстур и сэмплер со сравнением глубины ('<') для карты теней.
bool createSamplers() {
	auto& context = graphics::internal::context;

	const VkSamplerCreateInfo albedo_sampler = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.anisotropyEnable = VK_FALSE,
		.maxAnisotropy = 1.0f,
		.minLod = 0.0f,
		.maxLod = VK_LOD_CLAMP_NONE,
		.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
	};

	if (vkCreateSampler(context.device, &albedo_sampler, nullptr, &vk_sampler) != VK_SUCCESS) {
		std::cerr << "Failed to create albedo sampler\n";
		return false;
	}

	const VkFilter shadow_filter = shadow_linear_filter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;

	// Сэмплер карты теней: включено сравнение глубины с оператором '<' (VK_COMPARE_OP_LESS)
	const VkSamplerCreateInfo shadow_sampler = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = shadow_filter,
		.minFilter = shadow_filter,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.anisotropyEnable = VK_FALSE,
		.maxAnisotropy = 1.0f,
		.compareEnable = VK_TRUE,
		.compareOp = VK_COMPARE_OP_LESS,
		.minLod = 0.0f,
		.maxLod = 1.0f,
		.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
	};

	if (vkCreateSampler(context.device, &shadow_sampler, nullptr,
						&vk_shadow_sampler) != VK_SUCCESS) {
		std::cerr << "Failed to create shadow comparison sampler\n";
		return false;
	}

	return true;
}

// Наборы дескрипторов и конвейеры

bool createDescriptors() {
	auto& context = graphics::internal::context;

	// Основной набор 0: uniform сцены, два storage-буфера источников и карта теней (sampler2DShadow)
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
		{
			.binding = 3,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	// Основной набор 1: динамический uniform моделей и текстура альбедо
	const VkDescriptorSetLayoutBinding material_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	// Теневой проход: набор 0 (матрица солнца) и набор 1 (динамический буфер моделей)
	const VkDescriptorSetLayoutBinding shadow_scene_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
	};

	const VkDescriptorSetLayoutBinding shadow_model_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo scene_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(scene_bindings) / sizeof(scene_bindings[0]),
		.pBindings = scene_bindings,
	};

	const VkDescriptorSetLayoutCreateInfo material_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(material_bindings) / sizeof(material_bindings[0]),
		.pBindings = material_bindings,
	};

	const VkDescriptorSetLayoutCreateInfo shadow_scene_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &shadow_scene_binding,
	};

	const VkDescriptorSetLayoutCreateInfo shadow_model_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &shadow_model_binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &scene_layout, nullptr,
									&vk_scene_set_layout) != VK_SUCCESS ||
		vkCreateDescriptorSetLayout(context.device, &material_layout, nullptr,
									&vk_material_set_layout) != VK_SUCCESS ||
		vkCreateDescriptorSetLayout(context.device, &shadow_scene_layout, nullptr,
									&vk_shadow_scene_set_layout) != VK_SUCCESS ||
		vkCreateDescriptorSetLayout(context.device, &shadow_model_layout, nullptr,
									&vk_shadow_model_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layouts\n";
		return false;
	}

	const VkDescriptorPoolSize pool_sizes[] = {
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, uint32_t(1 + texture_count) },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, uint32_t(1 + texture_count) },
	};

	const VkDescriptorPoolCreateInfo pool = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = uint32_t(3 + texture_count),
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

	allocate.pSetLayouts = &vk_shadow_scene_set_layout;
	if (vkAllocateDescriptorSets(context.device, &allocate,
								 &vk_shadow_scene_descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate shadow scene descriptor set\n";
		return false;
	}

	allocate.pSetLayouts = &vk_shadow_model_set_layout;
	if (vkAllocateDescriptorSets(context.device, &allocate,
								 &vk_shadow_model_descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate shadow model descriptor set\n";
		return false;
	}

	allocate.pSetLayouts = &vk_material_set_layout;

	for (Texture& texture : textures) {
		if (vkAllocateDescriptorSets(context.device, &allocate,
									 &texture.descriptor_set) != VK_SUCCESS) {
			std::cerr << "Failed to allocate material descriptor set\n";
			return false;
		}
	}

	// Заполнение основного набора сцены (привязки 0..2 — буферы, привязка 3 — карта теней)
	const VkDescriptorBufferInfo scene_buffers[] = {
		{ scene_uniform.buffer, 0, VK_WHOLE_SIZE },
		{ point_light_storage.buffer, 0, VK_WHOLE_SIZE },
		{ spot_light_storage.buffer, 0, VK_WHOLE_SIZE },
	};

	const VkDescriptorImageInfo shadow_image_info = {
		.sampler = vk_shadow_sampler,
		.imageView = vk_shadow_image_view,
		.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	};

	const VkWriteDescriptorSet scene_writes[] = {
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = vk_scene_descriptor_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &scene_buffers[0],
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = vk_scene_descriptor_set,
			.dstBinding = 1,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &scene_buffers[1],
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = vk_scene_descriptor_set,
			.dstBinding = 2,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &scene_buffers[2],
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = vk_scene_descriptor_set,
			.dstBinding = 3,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = &shadow_image_info,
		},
	};

	vkUpdateDescriptorSets(context.device,
						   sizeof(scene_writes) / sizeof(scene_writes[0]), scene_writes, 0,
						   nullptr);

	// Заполнение наборов для прохода генерации карты теней
	const VkDescriptorBufferInfo model_buffer = { model_uniform.buffer, 0, sizeof(ModelData) };

	const VkWriteDescriptorSet shadow_writes[] = {
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = vk_shadow_scene_descriptor_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &scene_buffers[0],
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = vk_shadow_model_descriptor_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.pBufferInfo = &model_buffer,
		},
	};

	vkUpdateDescriptorSets(context.device,
						   sizeof(shadow_writes) / sizeof(shadow_writes[0]), shadow_writes, 0,
						   nullptr);

	for (const Texture& texture : textures) {
		const VkDescriptorImageInfo image = {
			.sampler = vk_sampler,
			.imageView = texture.view,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		};

		const VkWriteDescriptorSet writes[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = texture.descriptor_set,
				.dstBinding = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.pBufferInfo = &model_buffer,
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = texture.descriptor_set,
				.dstBinding = 1,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &image,
			},
		};

		vkUpdateDescriptorSets(context.device, 2, writes, 0, nullptr);
	}

	return true;
}

// Создаёт конвейер генерации карты теней: Dynamic Rendering (без VkRenderPass)
// и единственный вершинный шейдер, проецирующий геометрию в пространство солнца.
bool createShadowPipeline() {
	auto& context = graphics::internal::context;

	VkShaderModule vertex_shader = createShaderModule("shaders/shadow.vert.spv");
	if (vertex_shader == VK_NULL_HANDLE) {
		return false;
	}

	const VkDescriptorSetLayout set_layouts[] = {
		vk_shadow_scene_set_layout,
		vk_shadow_model_set_layout,
	};

	const VkPipelineLayoutCreateInfo pipeline_layout = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = sizeof(set_layouts) / sizeof(set_layouts[0]),
		.pSetLayouts = set_layouts,
	};

	if (vkCreatePipelineLayout(context.device, &pipeline_layout, nullptr,
							   &vk_shadow_pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create shadow pipeline layout\n";
		vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		return false;
	}

	const VkPipelineShaderStageCreateInfo vertex_stage = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.stage = VK_SHADER_STAGE_VERTEX_BIT,
		.module = vertex_shader,
		.pName = "main",
	};

	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription position_attribute = {
		.location = 0,
		.binding = 0,
		.format = VK_FORMAT_R32G32B32_SFLOAT,
		.offset = offsetof(Vertex, position),
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = 1,
		.pVertexAttributeDescriptions = &position_attribute,
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
		.depthBiasEnable = VK_TRUE,
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

	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 0,
		.pAttachments = nullptr,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
		VK_DYNAMIC_STATE_DEPTH_BIAS,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
		.pDynamicStates = dynamic_states,
	};

	// Описание форматов вложений для Dynamic Rendering (только буфер глубины)
	const VkPipelineRenderingCreateInfo rendering_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.colorAttachmentCount = 0,
		.pColorAttachmentFormats = nullptr,
		.depthAttachmentFormat = vk_shadow_format,
		.stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
	};

	const VkGraphicsPipelineCreateInfo pipeline = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = &rendering_info,
		.stageCount = 1,
		.pStages = &vertex_stage,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic_state,
		.layout = vk_shadow_pipeline_layout,
		.renderPass = VK_NULL_HANDLE,
		.subpass = 0,
	};

	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
													  &pipeline, nullptr, &vk_shadow_pipeline);

	vkDestroyShaderModule(context.device, vertex_shader, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create shadow graphics pipeline\n";
		return false;
	}

	return true;
}

// Создаёт основной графический конвейер (освещение, текстуры и наложение карты теней).
bool createPipeline() {
	auto& context = graphics::internal::context;

	VkShaderModule vertex_shader = createShaderModule("shaders/shadowed.vert.spv");
	VkShaderModule fragment_shader = createShaderModule("shaders/shadowed.frag.spv");

	if (vertex_shader == VK_NULL_HANDLE || fragment_shader == VK_NULL_HANDLE) {
		vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		return false;
	}

	const VkDescriptorSetLayout set_layouts[] = {
		vk_scene_set_layout,
		vk_material_set_layout,
	};

	const VkPushConstantRange push_constants = {
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(EffectConstants),
	};

	const VkPipelineLayoutCreateInfo pipeline_layout = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = sizeof(set_layouts) / sizeof(set_layouts[0]),
		.pSetLayouts = set_layouts,
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
			.offset = offsetof(Vertex, normal),
		},
		{
			.location = 2,
			.binding = 0,
			.format = VK_FORMAT_R32G32_SFLOAT,
			.offset = offsetof(Vertex, uv),
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

// Интерфейс

void drawPointLights() {
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

void drawTextureCombo(Model& model) {
	if (ImGui::BeginCombo("Texture", textures[model.texture].name)) {
		for (size_t i = 0; i < texture_count; ++i) {
			const bool selected = model.texture == int(i);

			if (ImGui::Selectable(textures[i].name, selected)) {
				model.texture = int(i);
			}

			if (selected) {
				ImGui::SetItemDefaultFocus();
			}
		}

		ImGui::EndCombo();
	}
}

void drawModels() {
	for (size_t i = 0; i < model_count; ++i) {
		Model& model = models[i];

		ImGui::PushID(int(i));

		const ImGuiTreeNodeFlags flags = i == 0 ? ImGuiTreeNodeFlags_DefaultOpen : 0;

		if (ImGui::TreeNodeEx("##model", flags, "%s", model.name)) {
			drawTextureCombo(model);

			ImGui::DragFloat3("Position", &model.position.x, 0.05f);

			if (model.mesh == mesh_octahedron) {
				ImGui::SliderFloat("Scale", &model.scale, 0.1f, 3.0f);
				ImGui::SliderFloat("Spin speed", &model.spin_speed, -180.0f, 180.0f,
								   "%.0f deg/s");
			}

			ImGui::ColorEdit3("Specular color", &model.specular.x);
			ImGui::SliderFloat("Shininess", &model.shininess, 1.0f, 256.0f, "%.0f",
							   ImGuiSliderFlags_Logarithmic);

			ImGui::Checkbox("Animated sampling", &model.animated);

			ImGui::BeginDisabled(!model.animated);
			ImGui::SliderFloat("Effect speed", &model.effect_speed, 0.0f, 5.0f);
			ImGui::SliderFloat("Distortion", &model.effect_distortion, 0.0f, 0.1f, "%.3f");
			ImGui::EndDisabled();

			ImGui::TreePop();
		}

		ImGui::PopID();
	}
}

void drawInterface() {
	ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(400.0f, 660.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Lab 4 - Shadow mapping");
	ImGui::PushItemWidth(-135.0f);

	ImGui::Text("%.1f FPS | Shadow map %ux%u", double(ImGui::GetIO().Framerate),
				shadow_map_size, shadow_map_size);

	if (ImGui::CollapsingHeader("Shadow mapping", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::SliderFloat("Ortho size", &shadow_ortho_size, 4.0f, 40.0f, "%.1f");
		ImGui::Text("Ortho area: %.1f x %.1f = %.0f", double(shadow_ortho_size),
					double(shadow_ortho_size), double(shadow_ortho_size * shadow_ortho_size));
		ImGui::SliderFloat("Depth bias", &shadow_depth_bias, 0.0f, 0.02f, "%.5f");
		ImGui::SliderFloat("Slope bias", &shadow_slope_bias, 0.0f, 8.0f, "%.2f");
		ImGui::Checkbox("Soft shadows (PCF)", &shadow_pcf);
	}

	if (ImGui::CollapsingHeader("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Direction##sun", &sun_direction.x, 0.01f, -1.0f, 1.0f);
		ImGui::ColorEdit3("Color##sun", &sun_color.x);
		ImGui::SliderFloat("Intensity##sun", &sun_intensity, 0.0f, 5.0f);
	}

	if (ImGui::CollapsingHeader("Camera")) {
		ImGui::DragFloat3("Position##camera", &camera_position.x, 0.05f);
		ImGui::SliderFloat("Yaw", &camera_yaw, -180.0f, 180.0f, "%.0f deg");
		ImGui::SliderFloat("Pitch", &camera_pitch, -89.0f, 89.0f, "%.0f deg");
		ImGui::SliderFloat("Field of view", &camera_fov, 20.0f, 120.0f, "%.0f deg");
	}

	if (ImGui::CollapsingHeader("Ambient light")) {
		ImGui::ColorEdit3("Color##ambient", &ambient_color.x);
		ImGui::SliderFloat("Intensity##ambient", &ambient_intensity, 0.0f, 1.0f);
	}

	if (ImGui::CollapsingHeader("Point lights")) {
		drawPointLights();
	}

	if (ImGui::CollapsingHeader("Spot lights")) {
		drawSpotLights();
	}

	if (ImGui::CollapsingHeader("Models")) {
		drawModels();
	}

	ImGui::PopItemWidth();
	ImGui::End();
}

// Переписывает буферы сцены, источников и моделей актуальными данными (включая матрицу солнца).
void uploadFrameData() {
	auto& context = graphics::internal::context;

	const float width = float(context.swapchain_extent.width);
	const float height = float(context.swapchain_extent.height);
	const float aspect = height > 0.0f ? width / height : 1.0f;

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

	// Матрица вида и ортографической проекции солнца (LookAt + orthographic)
	const Vec3 sun_dir = direction(sun_direction);
	const Vec3 light_target = { 0.0f, 0.0f, 0.0f };
	const Vec3 light_position = light_target - sun_dir * 18.0f;
	const Vec3 light_up = std::abs(sun_dir.y) > 0.99f ? Vec3{ 0.0f, 0.0f, -1.0f }
													  : Vec3{ 0.0f, 1.0f, 0.0f };

	const Mat4 light_view = math::lookAt(light_position, light_target, light_up);
	const Mat4 light_projection = math::orthographic(shadow_ortho_size, 1.0f, 1.0f, 40.0f);

	SceneData scene = {};
	scene.view_projection = projection * view;
	scene.light_view_projection = light_projection * light_view;
	store(scene.camera_position, camera_position, 1.0f);
	store(scene.ambient, ambient_color * ambient_intensity, 0.0f);
	store(scene.light_direction, sun_dir, 0.0f);
	store(scene.light_color, sun_color, sun_intensity);
	scene.point_light_count = uint32_t(point_lights.size());
	scene.spot_light_count = uint32_t(spot_lights.size());
	scene.shadow_bias = shadow_depth_bias;
	scene.shadow_pcf = shadow_pcf ? 1u : 0u;

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
		gpu_spots[i].cutoff[0] = std::cos(math::radians(light.inner_angle));
		gpu_spots[i].cutoff[1] = std::cos(math::radians(light.outer_angle));
	}

	upload(spot_light_storage, gpu_spots, spot_lights.size() * sizeof(GpuSpotLight));

	auto* const bytes = static_cast<uint8_t*>(model_uniform.data);

	for (size_t i = 0; i < model_count; ++i) {
		const Model& model = models[i];
		const float angle = math::radians(model.rotation + model.spin_speed * current_time);

		ModelData data = {};
		data.transform = math::translation(model.position) * math::rotationY(angle) *
						 math::scaling({ model.scale, model.scale, model.scale });
		store(data.specular, model.specular, model.shininess);

		std::memcpy(bytes + i * model_stride, &data, sizeof(data));
	}

	vmaFlushAllocation(context.allocator, model_uniform.allocation, 0, VK_WHOLE_SIZE);
}

// Проход внеэкранной генерации карты теней при помощи Dynamic Rendering (до основного прохода).
void renderShadowMap(VkCommandBuffer command_buffer) {
	if (vk_shadow_pipeline == VK_NULL_HANDLE) {
		return;
	}

	// Обновляем uniform-буферы один раз в начале кадра — общие данные для обоих проходов
	uploadFrameData();

	// Перевод карты теней в раскладку вложения глубины (предыдущее содержимое очистится)
	const VkImageMemoryBarrier to_depth = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = 0,
		.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
						 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = vk_shadow_image,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};

	vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
						 VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
							 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
						 0, 0, nullptr, 0, nullptr, 1, &to_depth);

	const VkRenderingAttachmentInfo depth_attachment = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = vk_shadow_image_view,
		.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.clearValue = { .depthStencil = { .depth = 1.0f, .stencil = 0 } },
	};

	const VkRenderingInfo rendering = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.renderArea = {
			.offset = { 0, 0 },
			.extent = { shadow_map_size, shadow_map_size },
		},
		.layerCount = 1,
		.colorAttachmentCount = 0,
		.pColorAttachments = nullptr,
		.pDepthAttachment = &depth_attachment,
	};

	vkCmdBeginRendering(command_buffer, &rendering);

	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = float(shadow_map_size),
		.height = float(shadow_map_size),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};

	const VkRect2D scissor = {
		.offset = { 0, 0 },
		.extent = { shadow_map_size, shadow_map_size },
	};

	vkCmdSetViewport(command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(command_buffer, 0, 1, &scissor);
	vkCmdSetDepthBias(command_buffer, 0.0f, 0.0f, shadow_slope_bias);

	vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_shadow_pipeline);

	vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
							vk_shadow_pipeline_layout, 0, 1, &vk_shadow_scene_descriptor_set,
							0, nullptr);

	for (size_t i = 0; i < model_count; ++i) {
		const Mesh& mesh = meshes[models[i].mesh];
		const uint32_t dynamic_offset = uint32_t(i * model_stride);

		vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
								vk_shadow_pipeline_layout, 1, 1,
								&vk_shadow_model_descriptor_set, 1, &dynamic_offset);

		const VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(command_buffer, 0, 1, &mesh.vertices.buffer, &offset);
		vkCmdBindIndexBuffer(command_buffer, mesh.indices.buffer, 0, VK_INDEX_TYPE_UINT16);

		vkCmdDrawIndexed(command_buffer, mesh.index_count, 1, 0, 0, 0);
	}

	vkCmdEndRendering(command_buffer);

	// Перевод карты теней в раскладку для чтения основным фрагментным шейдером
	const VkImageMemoryBarrier to_sample = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
		.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = vk_shadow_image,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};

	vkCmdPipelineBarrier(command_buffer,
						 VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
							 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
						 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
						 &to_sample);
}

} // namespace

// Вызывается один раз при запуске: буферы, текстуры, карта теней, дескрипторы и оба конвейера.
bool initialize() {
	auto& context = graphics::internal::context;

	VkPhysicalDeviceProperties properties = {};
	vkGetPhysicalDeviceProperties(context.physical_device, &properties);

	const VkDeviceSize alignment = properties.limits.minUniformBufferOffsetAlignment;
	model_stride = (sizeof(ModelData) + alignment - 1) / alignment * alignment;

	if (!createMappedBuffer(sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
							&scene_uniform) ||
		!createMappedBuffer(sizeof(GpuPointLight) * max_lights,
							VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &point_light_storage) ||
		!createMappedBuffer(sizeof(GpuSpotLight) * max_lights,
							VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &spot_light_storage) ||
		!createMappedBuffer(model_stride * model_count, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
							&model_uniform)) {
		return false;
	}

	if (!beginUpload()) {
		return false;
	}

	const bool mipmaps = canGenerateMipmaps(VK_FORMAT_R8G8B8A8_SRGB);
	bool loaded = createMeshes();

	for (Texture& texture : textures) {
		loaded = loaded && loadTexture(texture, mipmaps);
	}

	if (!endUpload() || !loaded) {
		return false;
	}

	if (!createShadowMap() || !createSamplers() || !createDescriptors()) {
		return false;
	}

	if (!createShadowPipeline() || !createPipeline()) {
		return false;
	}

	graphics::internal::setPreRenderCallback(renderShadowMap);

	return true;
}

// Вызывается один раз при выходе: уничтожает всё, созданное в initialize.
void shutdown() {
	graphics::internal::setPreRenderCallback(nullptr);

	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	for (const Mesh& mesh : meshes) {
		destroyBuffer(mesh.vertices);
		destroyBuffer(mesh.indices);
	}

	for (const Texture& texture : textures) {
		vkDestroyImageView(context.device, texture.view, nullptr);
		vmaDestroyImage(context.allocator, texture.image, texture.allocation);
	}

	vkDestroyImageView(context.device, vk_shadow_image_view, nullptr);
	vmaDestroyImage(context.allocator, vk_shadow_image, vma_shadow_allocation);

	vkDestroySampler(context.device, vk_shadow_sampler, nullptr);
	vkDestroySampler(context.device, vk_sampler, nullptr);

	destroyBuffer(model_uniform);
	destroyBuffer(spot_light_storage);
	destroyBuffer(point_light_storage);
	destroyBuffer(scene_uniform);

	vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_shadow_model_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_shadow_scene_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_material_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_scene_set_layout, nullptr);

	vkDestroyPipeline(context.device, vk_shadow_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, vk_shadow_pipeline_layout, nullptr);

	vkDestroyPipeline(context.device, vk_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);
}

// Вызывается каждый кадр: запоминает время и рисует интерфейс.
void update(double time) {
	current_time = float(time);
	drawInterface();
}

// Основной проход: рисует сцену с текстурами, освещением и выборкой из готовой карты теней.
void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

	if (vk_pipeline == VK_NULL_HANDLE) {
		return;
	}

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

	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
							vk_pipeline_layout, 0, 1, &vk_scene_descriptor_set, 0, nullptr);

	for (size_t i = 0; i < model_count; ++i) {
		const Model& model = models[i];
		const Mesh& mesh = meshes[model.mesh];

		const uint32_t dynamic_offset = uint32_t(i * model_stride);

		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
								vk_pipeline_layout, 1, 1,
								&textures[model.texture].descriptor_set, 1, &dynamic_offset);

		const EffectConstants effect = {
			.enabled = model.animated ? 1u : 0u,
			.time = current_time,
			.speed = model.effect_speed,
			.distortion = model.effect_distortion,
		};

		vkCmdPushConstants(fd.command_buffer, vk_pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
						   0, sizeof(effect), &effect);

		const VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &mesh.vertices.buffer, &offset);
		vkCmdBindIndexBuffer(fd.command_buffer, mesh.indices.buffer, 0, VK_INDEX_TYPE_UINT16);

		vkCmdDrawIndexed(fd.command_buffer, mesh.index_count, 1, 0, 0, 0);
	}
}

} // namespace application
