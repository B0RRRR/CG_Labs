# Лабораторные работы по компьютерной графике

Vulkan 1.1, C++20, CMake.

## Структура

```
source/            
  main.cpp
  graphics_internal.cpp/.hpp
  application.hpp
labs/
  lab1/
    application.cpp    код лабораторной
    shaders/           исходники шейдеров (.vert, .frag)
    README.md          условие варианта
    report.pdf         отчёт
shaders/           
```

## Сборка и запуск

```bash
cmake --preset debug -DLAB=1        # конфигурация, LAB — номер лабораторной
cmake --build build-debug --parallel
./build-debug/vulkan-starter-app
```

Пресеты: `debug`, `release`,
`msvc-debug`, `msvc-release`.



## Шейдеры

Исходники кладутся в `labs/labN/shaders/`, компиляция подключается в `CMakeLists.txt`:

```cmake
compile_shader(example.vert)
compile_shader(example.frag)
```

Собранные `.spv` попадают в `shaders/` в корне, оттуда их и загружает код.
Нужен `glslc` из Vulkan SDK — без него шаги компиляции шейдеров просто пропускаются.

## Стартовый код

```bash
git fetch upstream && git merge upstream/master
```
