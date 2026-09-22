# Лабораторные работы по компьютерной графике

Vulkan 1.1, C++20, CMake. Каждая лабораторная — отдельная папка в `labs/`,
стартовый код движка общий и лежит в `source/`.

## Структура

```
source/            общий код: окно (GLFW), инициализация Vulkan, ImGui
  main.cpp
  graphics_internal.cpp/.hpp
  application.hpp
labs/
  lab1/
    application.cpp    код лабораторной
    shaders/           исходники шейдеров (.vert, .frag)
    README.md          условие варианта
    report.pdf         отчёт
shaders/           сюда попадают скомпилированные .spv (в git не хранятся)
```

## Требования

Компилятор с поддержкой C++20 (GCC 10+, Clang 10+, MSVC 2019+), Vulkan SDK и CMake 3.20+.
Зависимости (GLFW, vk-bootstrap, VMA, Dear ImGui) скачиваются при конфигурации через
CMake `FetchContent`, устанавливать их отдельно не нужно.

## Сборка и запуск

```bash
cmake --preset debug -DLAB=1        # конфигурация, LAB — номер лабораторной
cmake --build build-debug --parallel
./build-debug/vulkan-starter-app
```

`-DLAB=2` соберёт вторую лабораторную и так далее. Пресеты: `debug`, `release`,
`msvc-debug`, `msvc-release` (каталог сборки — `build-debug` или `build-release`).

**Запускать из корня проекта** — пути к шейдерам и ресурсам относительные.

## Где писать код

Весь код лабораторной — в `labs/labN/application.cpp`, четыре функции:

| Функция | Для чего |
| --- | --- |
| `initialize` | создание объектов Vulkan (каждому `vkCreate*` нужен парный `vkDestroy*`) |
| `shutdown` | уничтожение всего, что создано в `initialize` |
| `update` | интерфейс на ImGui и обновление данных (например, матриц) |
| `render` | запись команд рисования (`vkCmd*`) в буфер из `FrameData` |

Объекты Vulkan (`VkDevice`, `VkRenderPass`, `VmaAllocator`, размер окна и прочее)
доступны через `graphics::internal::context` — см. `source/graphics_internal.hpp`.

Включена валидация Vulkan: нарушения спецификации печатаются в консоль.

## Шейдеры

Исходники кладутся в `labs/labN/shaders/`, компиляция подключается в `CMakeLists.txt`:

```cmake
compile_shader(example.vert)
compile_shader(example.frag)
```

Собранные `.spv` попадают в `shaders/` в корне, оттуда их и загружает код.
Нужен `glslc` из Vulkan SDK — без него шаги компиляции шейдеров просто пропускаются.

## Стартовый код

Основан на [vulkan-starter-app](https://github.com/vladeemerr/vulkan-starter-app)
(Copyright 2026 Vladimir Bakharev, Apache License 2.0, см. `LICENSE` и `NOTICE`).
Обновления оттуда подтягиваются так:

```bash
git fetch upstream && git merge upstream/master
```
