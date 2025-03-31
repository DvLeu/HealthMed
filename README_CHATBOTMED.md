
# Documentación de Instalación y Uso - Chatbot Médico HealthMED

## 1. Requisitos del Sistema

Antes de instalar el sistema, asegúrate de contar con lo siguiente:

### Hardware
- Procesador: Intel Core i5 o superior
- Memoria RAM: 8 GB mínimo (se recomienda 16 GB para uso con modelos de lenguaje)
- Almacenamiento: 1 GB de espacio libre

### Software
- Sistema operativo: Windows 10/11, macOS o cualquier distribución de Linux moderna
- Python 3.10 (recomendado)
- Navegador web moderno (Chrome, Firefox, Edge)
- Conexión a Internet (para acceso a API de OpenAI)

## 2. Instalación del Proyecto

### 2.1 Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/chatbotmed.git
cd chatbotmed
```

### 2.2 Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # En Linux/macOS
venv\Scripts\activate   # En Windows
```

### 2.3 Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2.4 Estructura de archivos esperada

```
chatbotmed/
├── main.py
├── NLP_comentado.py
├── CHATBOTMED_KNN_METRICAS.py
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── datasets/
│   ├── Dataset_Enfermedades_Limpio.csv
│   └── Glosario_Medico.csv
├── registro_sesiones.csv
└── requirements.txt
```

## 3. Configuración de API Keys

Para utilizar el modelo de lenguaje GPT (vía OpenAI), crea una variable de entorno con tu API Key:

### Linux/macOS

```bash
export OPENAI_API_KEY="tu_api_key_aqui"
```

### Windows (CMD)

```cmd
set OPENAI_API_KEY="tu_api_key_aqui"
```

## 4. Ejecución de la Aplicación

### 4.1 Backend

Ejecuta el archivo `main.py` para levantar el servidor FastAPI:

```bash
uvicorn main:app --reload
```

Esto iniciará el servidor en `http://127.0.0.1:8000`

### 4.2 Frontend

Abre el archivo `index.html` ubicado en la carpeta `templates/` directamente desde el navegador. También puedes servirlo con un servidor local si deseas.

## 5. Uso General

1. En la página inicial, el usuario puede ingresar síntomas marcando o escribiendo texto libre.
2. Al enviar, el sistema realiza un diagnóstico preliminar usando KNN y lógica clínica.
3. El usuario puede pedir explicaciones adicionales del diagnóstico vía el glosario interactivo.
4. Las respuestas son generadas usando NLP y modelos de lenguaje, con contexto médico.

## 6. Observaciones Técnicas

- Las métricas de precisión y evaluación del sistema se registran en `registro_sesiones.csv`
- El módulo de NLP funciona sin necesidad de FAISS, cargando directamente el glosario como contexto estático.
- Toda la interacción se da vía peticiones entre el frontend y los endpoints definidos en FastAPI.

## 7. Despliegue en Producción (Opcional)

Para entornos reales, se recomienda:
- Ejecutar detrás de un servidor como Nginx
- Usar HTTPS con certificados SSL
- Activar autenticación en endpoints sensibles
- Separar entorno de desarrollo y producción
