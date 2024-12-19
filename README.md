
# Proyecto AutoML Otoño 2024

# RAG para Ciencia de Datos

## 1. Introducción

Para la clase de Chatbots e Inteligencia Artificial del ITAM (Otoño 2024), elegimos crear un RAG utilizando Llama 3.2-1b, disponible en HuggingFace. La intención del proyecto es crear un chatbot que pueda ayudar a estudiantes de la carrera de Ciencia de Datos con dudas principalmente de materias de probabilidad, estadística y programación. La intención es ampliar el proyecto a que incluya información de todas las materias de la carrera.

## 2. Datos del proyecto

El proyecto utiliza Llama 3.2-1b de HuggingFace y Streamlit para ejecutar una pequeña app.

Está hecho para soportar formatos `.pdf` y `.docx`, pero hay que ser cuidadosos con el tipo de `.pdf` utilizados.

## 3. Inicialización

Para replicar el proyecto, es necesario, en primer lugar, tener un archivo `.env`, con un `HUGGING_FACE_TOKEN` que tenga permiso de `Read access to contents of all public gated repos you can access`. A partir de allí, es necesario meter los archivos para el RAG en la carpeta `knowledge_files`. Por cómo está hecho el código, es posible que cada persona meta sus propios archivos. En caso de querer los archivos con los que contamos, favor de contactarnos.

# 3.1. Ambiente virtual: Conda

Para poder replicar el código, dejamos archivos `.yml` para dos distintos casos de uso: cuando tienes CUDA disponible (`environment_cuda.yml`) y cuando no (`environment.yml`). Recordemos que un ejemplo para instalar las dependencias a partir de un `.yml` es:

```bash
conda env create -f environment.yml
```

O, en caso de tener CUDA:

```bash
conda env create -f environment_cuda.yml
```

# 3.2. Ejecución del Código

Para correr el codigo, ya teniendo activo el ambiente virtual (en nuestro caso, llamado `rag_env`), se corre el siguiente comando:

```bash
streamlit run app.py
```

Esto ejecutará una aplicación en localhost:8501. La primera vez que se ejecuta el código creará la base de datos vectorizada para el RAG, y después se podrá acceder a esa información sin necesidad de volver a ejecutar todo.
