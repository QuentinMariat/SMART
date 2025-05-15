// Analyzer module
function initAnalyzer() {
    // DOM Elements
    const youtubeBtn = document.getElementById("youtube-btn");
    const twitterBtn = document.getElementById("twitter-btn");
    const urlIcon = document.getElementById("url-icon");
    const urlInput = document.getElementById("url-input");
    const analyzeBtn = document.getElementById("analyze-btn");
    const sampleBtn = document.getElementById("sample-btn");
    const resultsContainer = document.getElementById("results-container");
    const commentCount = document.getElementById("comment-count");
    const positivePercent = document.getElementById("positive-percent");
    const neutralPercent = document.getElementById("neutral-percent");
    const negativePercent = document.getElementById("negative-percent");
    const positiveBar = document.getElementById("positive-bar");
    const neutralBar = document.getElementById("neutral-bar");
    const negativeBar = document.getElementById("negative-bar");
    const commentsContainer = document.getElementById("comments-container");
    const showPositive = document.getElementById("show-positive");
    const showNegative = document.getElementById("show-negative");
    const showAll = document.getElementById("show-all");

    // Sample URLs
    const sampleYoutubeUrl = "https://www.youtube.com/watch?v=dQw4w9WgXcQ";
    const sampleTwitterUrl = "https://twitter.com/elonmusk/status/1234567890";

    // Current platform (default: YouTube)
    let currentPlatform = "youtube";

    // Platform selection
    youtubeBtn?.addEventListener("click", () => {
        currentPlatform = "youtube";
        youtubeBtn.classList.add("active");
        twitterBtn.classList.remove("active");
        urlIcon.className = "fab fa-youtube platform-icon";
        urlInput.placeholder = "Paste YouTube video URL here...";
    });

    twitterBtn?.addEventListener("click", () => {
        currentPlatform = "twitter";
        twitterBtn.classList.add("active");
        youtubeBtn.classList.remove("active");
        urlIcon.className = "fab fa-twitter platform-icon";
        urlInput.placeholder = "Paste Twitter/X post URL here...";
    });

    // Sample URL button
    sampleBtn?.addEventListener("click", () => {
        urlInput.value =
            currentPlatform === "youtube" ? sampleYoutubeUrl : sampleTwitterUrl;
    });

    // Analyze button
    analyzeBtn?.addEventListener("click", async () => {
        const url = urlInput.value.trim();

        if (!url) {
            alert("Merci d'entrer une URL à analyser");
            return;
        }

        // Validate URL format
        if (
            currentPlatform === "youtube" &&
            !url.includes("youtube.com/watch")
        ) {
            alert("Merci d'entrer une URL de vidéo YouTube valide");
            return;
        }

        if (currentPlatform === "twitter" && !url.includes("twitter.com/")) {
            alert("Merci d'entrer une URL de post Twitter valide");
            return;
        }

        // Show loading state
        analyzeBtn.innerHTML =
            '<i class="fas fa-spinner fa-spin mr-2"></i> Récupération et analyse des commentaires...';
        analyzeBtn.disabled = true;

        // Afficher un message d'attente
        const resultsContainer = document.getElementById("results-container");
        resultsContainer.classList.remove("hidden");

        // Montrer l'icône de chargement dans l'histogramme
        const emotionsContainer = document.getElementById("emotions-histogram");
        if (emotionsContainer) {
            emotionsContainer.innerHTML = `
                <div class="flex flex-col items-center justify-center h-full text-center">
                    <div class="animate-spin text-indigo-600 text-4xl mb-4">
                        <i class="fas fa-sync-alt"></i>
                    </div>
                    <h3 class="text-lg font-semibold text-indigo-700 mb-2">Analyse en cours...</h3>
                    <p class="text-gray-700 mb-3">
                        Récupération et analyse de TOUS les commentaires disponibles.
                        Cette opération peut prendre jusqu'à 2 minutes.
                    </p>
                </div>
            `;
        }

        // Simulate API call with timeout
        setTimeout(async () => {
            // For demo purposes, we'll generate mock data
            //const mockData = generateMockAnalysis();
            const mockData = await fetchAnalysisFromAPI(url, currentPlatform);

            console.log("étape 1");
            // Display results
            //displayResults(mockData);
            console.log("étape 2");
            // Reset button
            analyzeBtn.innerHTML =
                '<i class="fas fa-chart-bar mr-2"></i> Analyser les sentiments';
            console.log("étape 3");
            analyzeBtn.disabled = false;
            console.log("étape 4");
        }, 2000);
    });

    // Filter comments by sentiment
    showPositive?.addEventListener("click", () => {
        document.querySelectorAll(".comment").forEach(comment => {
            comment.style.display =
                comment.dataset.sentiment === "positive" ? "block" : "none";
        });
    });

    showNegative?.addEventListener("click", () => {
        document.querySelectorAll(".comment").forEach(comment => {
            comment.style.display =
                comment.dataset.sentiment === "negative" ? "block" : "none";
        });
    });

    showAll?.addEventListener("click", () => {
        document.querySelectorAll(".comment").forEach(comment => {
            comment.style.display = "block";
        });
    });
}

async function fetchAnalysisFromAPI(url, platform) {
    const endpoint =
        platform === "youtube"
            ? "http://localhost:8000/analyze/youtube"
            : "http://localhost:8000/analyze/twitter";

    try {
        // Définir un timeout pour la requête
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 secondes de timeout

        const response = await fetch(endpoint, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ url }),
            signal: controller.signal
        });

        // Nettoyer le timeout
        clearTimeout(timeoutId);

        if (!response.ok) {
            const errorData = await response
                .json()
                .catch(() => ({ error: "Erreur inconnue" }));
            throw new Error(
                errorData.error || `Erreur API: ${response.statusText}`
            );
        }

        const data = await response.json();

        // Créer des données d'émotions fictives si elles n'existent pas
        if (!data.emotion_counts) {
            console.log(
                "Aucune donnée d'émotions reçue du backend, création de données fictives"
            );
            // Convertir les pourcentages de sentiment en données d'émotions
            data.emotion_counts = {
                joy: Math.round(data.positive * 100 * 0.6),
                admiration: Math.round(data.positive * 100 * 0.4),
                disappointment: Math.round(data.negative * 100 * 0.5),
                annoyance: Math.round(data.negative * 100 * 0.3),
                anger: Math.round(data.negative * 100 * 0.2),
                neutral: Math.round(data.neutral * 100)
            };
        }

        // Afficher les résultats
        displayResults(data);
    } catch (error) {
        console.error("Erreur lors de l'analyse:", error);

        // Réinitialiser le bouton d'analyse
        const analyzeBtn = document.getElementById("analyze-btn");
        if (analyzeBtn) {
            analyzeBtn.innerHTML =
                '<i class="fas fa-chart-bar mr-2"></i> Analyser les sentiments';
            analyzeBtn.disabled = false;
        }

        // Afficher un message d'erreur à l'utilisateur
        const resultsContainer = document.getElementById("results-container");
        if (resultsContainer) {
            resultsContainer.classList.remove("hidden");

            // Afficher l'erreur dans l'histogramme
            const emotionsContainer =
                document.getElementById("emotions-histogram");
            if (emotionsContainer) {
                const errorMsg =
                    error.message ||
                    "Impossible d'analyser cette URL. Le serveur ne répond pas.";

                emotionsContainer.innerHTML = "";

                // Créer les éléments du message d'erreur
                const errorContainer = document.createElement("div");
                errorContainer.className =
                    "flex flex-col items-center justify-center h-full text-center";

                // Icône d'erreur
                const iconDiv = document.createElement("div");
                iconDiv.className = "text-red-500 text-4xl mb-4";
                const icon = document.createElement("i");
                icon.className = "fas fa-exclamation-triangle";
                iconDiv.appendChild(icon);

                // Titre d'erreur
                const errorTitle = document.createElement("h3");
                errorTitle.className =
                    "text-lg font-semibold text-red-700 mb-2";
                errorTitle.textContent = "Erreur lors de l'analyse";

                // Message d'erreur
                const errorText = document.createElement("p");
                errorText.className = "text-gray-700 mb-3";
                errorText.textContent = errorMsg;

                // Instructions
                const instructionsDiv = document.createElement("div");
                instructionsDiv.className = "text-sm text-gray-500";
                instructionsDiv.textContent = "Veuillez vérifier que:";

                // Liste des instructions
                const instructionsList = document.createElement("ul");
                instructionsList.className = "list-disc text-left pl-5 mt-2";

                const items = [
                    "Le serveur backend est en cours d'exécution",
                    "L'URL de la vidéo/post est correcte",
                    "La vidéo/post est accessible publiquement"
                ];

                items.forEach(item => {
                    const li = document.createElement("li");
                    li.textContent = item;
                    instructionsList.appendChild(li);
                });

                instructionsDiv.appendChild(instructionsList);

                // Assembler tous les éléments
                errorContainer.appendChild(iconDiv);
                errorContainer.appendChild(errorTitle);
                errorContainer.appendChild(errorText);
                errorContainer.appendChild(instructionsDiv);

                // Ajouter à la page
                emotionsContainer.appendChild(errorContainer);
            }

            // Masquer les autres sections qui nécessitent des données
            document.querySelectorAll(".sentiment-summary").forEach(el => {
                el.style.display = "none";
            });

            document.getElementById("comments-container").innerHTML =
                '<p class="text-gray-500 text-center p-4">Aucun commentaire disponible en raison d\'une erreur</p>';
        }

        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: "smooth" });
    }
}

// Generate mock analysis data
function generateMockAnalysis() {
    const totalComments = Math.floor(Math.random() * 500) + 100;
    const positive = Math.random() * 0.6;
    const negative = Math.random() * 0.3;
    const neutral = 1 - positive - negative;

    // Generate mock comments
    const comments = [];
    const positiveComments = [
        "This is amazing! Love the content.",
        "Great video, very informative!",
        "Best explanation I've seen on this topic.",
        "Keep up the good work!",
        "I've watched this 3 times already. So good!"
    ];

    const negativeComments = [
        "This is terrible, completely wrong information.",
        "Waste of time, didn't learn anything.",
        "The worst video on this topic.",
        "Disappointed with the quality.",
        "Not what I expected at all."
    ];

    const neutralComments = [
        "Interesting perspective.",
        "I'll need to research this more.",
        "The video quality is good.",
        "Not sure I agree with everything.",
        "Decent explanation overall."
    ];

    // Generate positive comments
    for (let i = 0; i < Math.floor(positive * 10); i++) {
        comments.push({
            text: positiveComments[
                Math.floor(Math.random() * positiveComments.length)
            ],
            sentiment: "positive",
            likes: Math.floor(Math.random() * 1000),
            time: `${Math.floor(Math.random() * 60)}:${Math.floor(
                Math.random() * 60
            )
                .toString()
                .padStart(2, "0")}`
        });
    }

    // Generate negative comments
    for (let i = 0; i < Math.floor(negative * 10); i++) {
        comments.push({
            text: negativeComments[
                Math.floor(Math.random() * negativeComments.length)
            ],
            sentiment: "negative",
            likes: Math.floor(Math.random() * 500),
            time: `${Math.floor(Math.random() * 60)}:${Math.floor(
                Math.random() * 60
            )
                .toString()
                .padStart(2, "0")}`
        });
    }

    // Generate neutral comments
    for (let i = 0; i < Math.floor(neutral * 10); i++) {
        comments.push({
            text: neutralComments[
                Math.floor(Math.random() * neutralComments.length)
            ],
            sentiment: "neutral",
            likes: Math.floor(Math.random() * 200),
            time: `${Math.floor(Math.random() * 60)}:${Math.floor(
                Math.random() * 60
            )
                .toString()
                .padStart(2, "0")}`
        });
    }

    // Shuffle comments
    comments.sort(() => Math.random() - 0.5);

    return {
        totalComments,
        positive: parseFloat(positive.toFixed(2)),
        negative: parseFloat(negative.toFixed(2)),
        neutral: parseFloat(neutral.toFixed(2)),
        comments
    };
}

// Display analysis results
function displayResults(data) {
    const resultsContainer = document.getElementById("results-container");
    const commentCount = document.getElementById("comment-count");
    const positivePercent = document.getElementById("positive-percent");
    const neutralPercent = document.getElementById("neutral-percent");
    const negativePercent = document.getElementById("negative-percent");
    const positiveBar = document.getElementById("positive-bar");
    const neutralBar = document.getElementById("neutral-bar");
    const negativeBar = document.getElementById("negative-bar");

    // Show results container
    resultsContainer.classList.remove("hidden");

    // Afficher un message de succès temporaire
    const successMsg = document.createElement("div");
    successMsg.className =
        "bg-green-100 text-green-800 p-3 rounded-lg mb-4 flex items-center";
    successMsg.innerHTML = `
        <i class="fas fa-check-circle text-green-600 mr-2 text-xl"></i>
        <span><strong>Analyse complète !</strong> Tous les ${data.totalComments} commentaires ont été analysés avec succès.</span>
    `;
    // Insérer au début du conteneur de résultats
    resultsContainer.insertBefore(successMsg, resultsContainer.firstChild);

    // Faire disparaître le message après 10 secondes
    setTimeout(() => {
        successMsg.style.transition = "opacity 1s";
        successMsg.style.opacity = 0;
        setTimeout(() => successMsg.remove(), 1000);
    }, 10000);

    // Update counts and percentages
    commentCount.textContent = data.totalComments;
    positivePercent.textContent = `${Math.round(data.positive * 100)}%`;
    neutralPercent.textContent = `${Math.round(data.neutral * 100)}%`;
    negativePercent.textContent = `${Math.round(data.negative * 100)}%`;

    // Update progress bars
    positiveBar.style.width = `${data.positive * 100}%`;
    neutralBar.style.width = `${data.neutral * 100}%`;
    negativeBar.style.width = `${data.negative * 100}%`;

    // Re-show sentiment summaries (in case they were hidden by an error)
    document.querySelectorAll(".sentiment-summary").forEach(el => {
        el.style.display = "block";
    });

    // Generate emotions histogram if emotion_counts exist
    if (data.emotion_counts) {
        renderEmotionsHistogram(data.emotion_counts);
    }

    // Display comments
    console.log("étape 1.1");
    renderComments(data.comments);
    console.log("étape 1.2");

    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: "smooth" });
    console.log("étape 1.3");
}

// Render the emotions histogram
function renderEmotionsHistogram(emotionCounts) {
    const emotionsContainer = document.getElementById("emotions-histogram");
    emotionsContainer.innerHTML = "";

    // If no emotions data
    if (!emotionCounts || Object.keys(emotionCounts).length === 0) {
        emotionsContainer.innerHTML =
            '<div class="flex justify-center items-center h-full"><span class="text-gray-500">Aucune donnée d\'émotion disponible</span></div>';
        return;
    }

    // Get the emotions sorted by count (highest first)
    const sortedEmotions = Object.entries(emotionCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 6); // Get top 6 emotions

    // Calculate total for percentages
    const totalCount = Object.values(emotionCounts).reduce(
        (sum, count) => sum + count,
        0
    );

    // Add title with total count
    const titleDiv = document.createElement("div");
    titleDiv.className = "text-lg font-semibold text-gray-800 mb-4";
    titleDiv.textContent = `Top 6 émotions détectées`;
    emotionsContainer.appendChild(titleDiv);

    // Create emotions list container
    const emotionsList = document.createElement("div");
    emotionsList.className = "space-y-4";

    // Traduction des émotions en français
    const emotionTranslations = {
        joy: "Joie",
        admiration: "Admiration",
        amusement: "Amusement",
        excitement: "Excitation",
        gratitude: "Gratitude",
        love: "Amour",
        optimism: "Optimisme",
        pride: "Fierté",
        relief: "Soulagement",
        approval: "Approbation",

        anger: "Colère",
        annoyance: "Agacement",
        disappointment: "Déception",
        disapproval: "Désapprobation",
        disgust: "Dégoût",
        embarrassment: "Embarras",
        fear: "Peur",
        grief: "Chagrin",
        remorse: "Remords",
        sadness: "Tristesse",

        neutral: "Neutre"
    };

    // Couleurs associées aux émotions
    const emotionColors = {
        joy: "#10B981", // green
        admiration: "#3B82F6", // blue
        amusement: "#8B5CF6", // purple
        excitement: "#EC4899", // pink
        gratitude: "#F59E0B", // amber
        love: "#EF4444", // red
        optimism: "#F97316", // orange
        pride: "#6366F1", // indigo
        relief: "#14B8A6", // teal
        approval: "#0EA5E9", // sky

        anger: "#DC2626", // red
        annoyance: "#F97316", // orange
        disappointment: "#78716C", // stone
        disapproval: "#9CA3AF", // gray
        disgust: "#65A30D", // lime
        embarrassment: "#EC4899", // pink
        fear: "#7C3AED", // violet
        grief: "#374151", // gray
        remorse: "#6B7280", // gray
        sadness: "#1D4ED8", // blue

        neutral: "#9CA3AF" // gray
    };

    // Créer un élément pour chaque émotion
    sortedEmotions.forEach(([emotion, count]) => {
        // Calculer le pourcentage arrondi
        const percentage = Math.round((count / totalCount) * 100);
        const emotionName = emotionTranslations[emotion] || emotion;

        // Créer un conteneur pour cette émotion
        const emotionItem = document.createElement("div");
        emotionItem.className = "w-full";

        // En-tête avec nom et pourcentage
        const header = document.createElement("div");
        header.className = "flex justify-between items-center mb-1";

        const nameElem = document.createElement("div");
        nameElem.className = "font-medium text-gray-800";
        nameElem.textContent = emotionName;

        const percentElem = document.createElement("div");
        percentElem.className = "font-semibold text-gray-900";
        percentElem.textContent = `${percentage}%`;

        header.appendChild(nameElem);
        header.appendChild(percentElem);
        emotionItem.appendChild(header);

        // Barre de progression
        const barContainer = document.createElement("div");
        barContainer.className = "w-full bg-gray-200 rounded-full h-4";

        const bar = document.createElement("div");
        bar.className = "h-4 rounded-full";
        bar.style.width = `${percentage}%`;
        bar.style.backgroundColor = emotionColors[emotion] || "#9CA3AF";

        barContainer.appendChild(bar);
        emotionItem.appendChild(barContainer);

        // Info additionnelle en petit
        const countInfo = document.createElement("div");
        countInfo.className = "text-xs text-gray-500 mt-1";
        emotionItem.appendChild(countInfo);

        // Ajouter à la liste
        emotionsList.appendChild(emotionItem);
    });

    emotionsContainer.appendChild(emotionsList);

    // Remarque explicative
    const note = document.createElement("div");
    note.className = "text-sm text-gray-600 mt-6 italic";
    note.textContent =
        "Les pourcentages représentent la proportion de chaque émotion parmi tous les commentaires.";
    emotionsContainer.appendChild(note);
}

// Render comments
function renderComments(comments) {
    const commentsContainer = document.getElementById("comments-container");
    commentsContainer.innerHTML = "";

    comments.forEach(comment => {
        const commentDiv = document.createElement("div");
        commentDiv.className = `comment p-3 rounded-lg border ${
            comment.sentiment === "positive"
                ? "border-green-200 bg-green-50"
                : comment.sentiment === "negative"
                ? "border-red-200 bg-red-50"
                : "border-gray-200 bg-gray-50"
        }`;
        commentDiv.dataset.sentiment = comment.sentiment;

        commentDiv.innerHTML = `
            <div class="flex justify-between items-start mb-1">
                <p class="text-sm font-medium ${
                    comment.sentiment === "positive"
                        ? "text-green-800"
                        : comment.sentiment === "negative"
                        ? "text-red-800"
                        : "text-gray-800"
                }">${comment.text}</p>
                <span class="text-xs text-gray-500 ml-2">${comment.time}</span>
            </div>
            <div class="flex justify-between items-center">
                <span class="text-xs ${
                    comment.sentiment === "positive"
                        ? "text-green-600"
                        : comment.sentiment === "negative"
                        ? "text-red-600"
                        : "text-gray-600"
                } capitalize">${comment.sentiment}</span>
                <div class="flex items-center text-xs text-gray-500">
                    <i class="fas fa-thumbs-up mr-1"></i>
                    <span>${comment.likes}</span>
                </div>
            </div>
        `;

        commentsContainer.appendChild(commentDiv);
    });
}
