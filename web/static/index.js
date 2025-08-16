const featuresData = [
  {
      title: "Easy Scheduling",
      description: "Book and manage your appointments without the hassle of long queues."
  },
  {
      title: "Health Reminders",
      description: "Never miss a medication dose or a doctor's visit again."
  },
  {
      title: "Secure Records",
      description: "Your medical history is encrypted and safe with us."
  }
];

const featuresSection = document.getElementById("featuresSection");

featuresData.forEach(feature => {
  const card = document.createElement("div");
  card.classList.add("feature-card");
  card.innerHTML = `
    <h3>${feature.title}</h3>
    <p>${feature.description}</p>
  `;
  featuresSection.appendChild(card);
});

// Navigate to selection page
document.getElementById("startBtn").addEventListener("click", () => {
  window.location.href = "explore.html";
});

const notify = (feature) => alert(`Opening ${feature} (coming soon)`);
  document.getElementById('selectChatbot').addEventListener('click', (e) => { e.preventDefault(); notify('AI Chatbot'); });