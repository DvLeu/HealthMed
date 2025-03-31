// Referencia al contenedor de mensajes del chat
const chatBox = document.getElementById("chat-box");

/**
 * Convierte texto en formato Markdown simple (como **negritas**) y saltos de línea (\n) en HTML.
 * @param {string} text - Texto a convertir
 * @returns {string} - HTML resultante
 */
function parseMarkdown(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // convierte **texto** en <strong>texto</strong>
    .replace(/\n/g, '<br>'); // convierte \n en <br> para salto de línea en HTML
}

/**
 * Añade un mensaje al chat visualmente.
 * @param {string} text - El contenido del mensaje
 * @param {string} sender - Quién envía el mensaje ('user' o 'bot')
 */
function appendMessage(text, sender = "bot") {
  // Crear contenedor principal del mensaje
  const messageDiv = document.createElement("div");
  messageDiv.classList.add("message", sender);

  // Crear avatar con iniciales o etiqueta
  const avatar = document.createElement("div");
  avatar.classList.add("avatar");
  avatar.textContent = sender === "user" ? "Tú" : "AI";

  // Crear burbuja de mensaje con el contenido
  const bubble = document.createElement("div");
  bubble.classList.add("bubble");
  bubble.innerHTML = parseMarkdown(text); // Aplica formato Markdown

  // Añadir avatar y burbuja al contenedor del mensaje
  messageDiv.appendChild(avatar);
  messageDiv.appendChild(bubble);

  // Agregar el mensaje al chat y hacer scroll automático al final
  chatBox.appendChild(messageDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
}

/**
 * Función principal que se ejecuta cuando el usuario hace una pregunta.
 * Envía la pregunta a la API y muestra la respuesta.
 */
async function consultarAPI() {
  const input = document.getElementById("pregunta");
  const pregunta = input.value.trim();
  if (!pregunta) return; // Si está vacío, no hace nada

  // Muestra el mensaje del usuario
  appendMessage(pregunta, "user");
  input.value = ""; // Limpia el campo de entrada

  // Muestra un mensaje temporal mientras se consulta la API
  appendMessage("Escribiendo respuesta...", "bot");

  try {
    // Envío de la solicitud POST a la API con la pregunta
    const res = await fetch("https://healthmed-api-nlp.onrender.com/preguntar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pregunta }) // Enviar pregunta como JSON
    });

    // Procesa la respuesta de la API
    const data = await res.json();

    // Reemplaza el mensaje temporal con la respuesta formateada
    const allMessages = document.querySelectorAll(".bot .bubble");
    const lastBubble = allMessages[allMessages.length - 1];
    lastBubble.innerHTML = parseMarkdown(data.respuesta);

  } catch (error) {
    // En caso de error, reemplaza con mensaje de error
    const allMessages = document.querySelectorAll(".bot .bubble");
    const lastBubble = allMessages[allMessages.length - 1];
    lastBubble.textContent = "Error al consultar la API 😢";
  }
}
