# Лабораторная работа №4: Наложение теней

**Объект: правильный октаэдр** в комнате с полом и тремя стенами, текстурированием и освещением по модели Блинна-Фонга из третьей работы, дополненный картой теней от направленного источника света (солнца).

Выполнено базовое задание (на оценку 4).

## Сборка и запуск

Из корня репозитория:

```bash
cmake --preset debug -DLAB=4
cmake --build build-debug --parallel
./build-debug/vulkan-starter-app
```

Программу нужно запускать из корня репозитория: шейдеры читаются из `shaders/`,
текстуры — из `labs/lab3/textures/`.

## Что реализовано

| Требование | Где в коде |
| --- | --- |
| Vulkan 1.3 и включение `dynamicRendering` | `source/graphics_internal.cpp` (`initialize`) |
| Внеэкранная генерация карты теней каждый кадр через Dynamic Rendering (`vkCmdBeginRendering` / `vkCmdEndRendering`, без `VkRenderPass` и `VkFramebuffer`) | `renderShadowMap`, `createShadowMap` в `labs/lab4/application.cpp` |
| Отдельный графический конвейер для карты теней с `VkPipelineRenderingCreateInfo` и единственным вершинным шейдером | `createShadowPipeline`, `labs/lab4/shaders/shadow.vert` |
| Матрица солнца: `LookAt` + ортографическая проекция | `uploadFrameData` в `labs/lab4/application.cpp` |
| Сэмплер со сравнением глубины (`compareEnable = VK_TRUE`, `VK_COMPARE_OP_LESS`) | `createSamplers` в `labs/lab4/application.cpp` |
| Ресурс `sampler2DShadow` в основном фрагментном шейдере и расчёт затенения (с опциональным 3×3 PCF) | `calculateShadow` в `labs/lab4/shaders/shadowed.frag` |
| Элементы интерфейса: площадь ортографической проекции и сдвиг глубины (`Depth bias`, `Slope bias`) | `drawInterface` в `labs/lab4/application.cpp` |

## Файлы

```
application.cpp         карта теней, Dynamic Rendering конвейер, сэмплер сравнения, интерфейс
math.hpp                векторы и матрицы (LookAt, perspective, orthographic)
shaders/shadow.vert     единственный шейдер теневого прохода (проекция в пространство солнца)
shaders/shadowed.vert   основной вершинный шейдер (передача координат в пространстве солнца)
shaders/shadowed.frag   основной фрагментный шейдер (sampler2DShadow + Блинн-Фонг + текстуры)
```
