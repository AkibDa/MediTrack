const featuresData = [
  {
      title: "Symptom Checker",
      description: "Get instant insights about possible health conditions based on your symptoms.",
      icon: "🔍",
      action: "Check Symptoms",
      url: "/symptom_checker"
  },
  {
      title: "Find Doctors",
      description: "Locate healthcare professionals and specialists in your area.",
      icon: "👨‍⚕️",
      action: "Find Doctors",
      url: "/doctors"
  },
  {
      title: "Health Reminders",
      description: "Set and manage medication and appointment reminders.",
      icon: "⏰",
      action: "Set Reminders",
      url: "/reminders"
  },
  {
      title: "AI Health Assistant",
      description: "Get instant answers to your health questions from our AI chatbot.",
      icon: "🤖",
      action: "Chat Now",
      url: "/chatbot"
  }
];

const featuresSection = document.getElementById("featuresSection");

if (featuresSection) {
  // Add a header showing the number of available features
  const featuresHeader = document.createElement("div");
  featuresHeader.style.cssText = "text-align: center; margin-bottom: 2rem; color: white;";
  featuresHeader.innerHTML = `<p>Discover ${featuresData.length} powerful health management tools</p>`;
  featuresSection.appendChild(featuresHeader);
  
  featuresData.forEach((feature, index) => {
    const card = document.createElement("div");
    card.classList.add("feature-card");
    card.innerHTML = `
      <div class="feature-icon">${feature.icon}</div>
      <h3>${feature.title}</h3>
      <p>${feature.description}</p>
      <a href="${feature.url}" class="feature-action-btn">${feature.action}</a>
    `;
    
    // Add click event to the entire card
    card.addEventListener('click', () => {
      // Add visual feedback
      card.style.transform = 'scale(0.95)';
      card.style.transition = 'transform 0.1s ease';
      
      // Reset transform after animation
      setTimeout(() => {
        card.style.transform = 'scale(1)';
        card.style.transition = 'transform 0.3s ease';
      }, 100);
      
      // Navigate to the feature
      setTimeout(() => {
        window.location.href = feature.url;
      }, 150);
    });
    
    // Add hover effect for the action button
    const actionBtn = card.querySelector('.feature-action-btn');
    if (actionBtn) {
      actionBtn.addEventListener('mouseenter', () => {
        actionBtn.style.transform = 'translateY(-2px) scale(1.05)';
      });
      
      actionBtn.addEventListener('mouseleave', () => {
        actionBtn.style.transform = 'translateY(0) scale(1)';
      });
    }
    
    featuresSection.appendChild(card);
  });
}

// Navigate to selection page
const startBtn = document.getElementById("startBtn");
if (startBtn) {
  // Check if user is logged in
  const isLoggedIn = document.querySelector('a[href="/logout"]') !== null;
  
  if (isLoggedIn) {
    // User is logged in, go to explore page
    startBtn.addEventListener("click", () => {
      window.location.href = "/explore";
    });
  } else {
    // User is not logged in, go to login page
    startBtn.textContent = "Login to Get Started";
    startBtn.addEventListener("click", () => {
      window.location.href = "/login";
    });
  }
}

const notify = (feature) => alert(`Opening ${feature} (coming soon)`);
const selectChatbot = document.getElementById('selectChatbot');
if (selectChatbot) {
  selectChatbot.addEventListener('click', (e) => { 
    e.preventDefault(); 
    notify('AI Chatbot'); 
  });
}