const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
let chatHistory = [];

function addMessage(text, sender) {
  const messageDiv = document.createElement('div');
  messageDiv.className = sender === 'user' ? 'flex justify-end' : 'flex justify-start';

  const bubbleColor = sender === 'user' ? 'bg-green-600' : 'bg-gray-700';
  const bubble = document.createElement('div');
  bubble.className = `${bubbleColor} px-4 py-2 rounded-2xl max-w-xs text-sm break-words`;

  bubble.innerText = text;
  messageDiv.appendChild(bubble);
  chatMessages.appendChild(messageDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage() {
  const message = messageInput.value.trim();
  if (!message) return;

  addMessage(message, 'user');
  messageInput.value = '';

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history: chatHistory })
    });

    const data = await res.json();

    let displayText = '';
    if (data.reply) {
      displayText = data.reply;
    } else {
      displayText = JSON.stringify(data, null, 2);
    }

    addMessage(displayText, 'bot');

    chatHistory.push({
      role: 'user',
      parts: [{ text: message }]
    });
    chatHistory.push({
      role: 'model',
      parts: [{ text: displayText }]
    });
  } catch (error) {
    addMessage('❌ Error connecting to server.', 'bot');
    console.error(error);
  }
}
