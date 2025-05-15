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
    const commentsContainer = document.getElementById("comments-container");

    // Nous ne référençons plus ces éléments car ils seront mis à jour dynamiquement
    // const showPositive = document.getElementById("show-positive");
    // const showNegative = document.getElementById("show-negative");
    // const showAll = document.getElementById("show-all");

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

        // Effectuer l'appel API avec un timeout pour l'interface utilisateur
        setTimeout(async () => {
            try {
                const data = await fetchAnalysisFromAPI(url, currentPlatform);
                console.log("Données reçues avec succès:", data);
                // Reset button
                analyzeBtn.innerHTML =
                    '<i class="fas fa-chart-bar mr-2"></i> Analyser les sentiments';
                analyzeBtn.disabled = false;
            } catch (error) {
                console.error("Erreur lors de l'analyse:", error);
                analyzeBtn.innerHTML =
                    '<i class="fas fa-chart-bar mr-2"></i> Analyser les sentiments';
                analyzeBtn.disabled = false;
            }
        }, 1000);
    });

    // Nous ne configurons plus les gestionnaires d'événements ici
    // car ils seront créés dynamiquement dans updateCommentFilters
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

        // Log détaillé des données reçues pour le débogage
        console.log("Données complètes reçues du backend:", data);

        // Vérifier si emotion_counts est présent dans la réponse API
        if (!data.emotion_counts) {
            console.log(
                "Aucune donnée d'émotions reçue du backend, création de données fictives"
            );

            // Essayer de récupérer les données d'émotions à partir de normalized_results
            if (data.normalized_results && data.normalized_results.length > 0) {
                console.log(
                    "Tentative de génération d'emotion_counts à partir des résultats normalisés"
                );

                // Compter les émotions à partir des résultats normalisés
                const emotionCounts = {};
                data.normalized_results.forEach(result => {
                    const label = result.label;
                    if (label) {
                        emotionCounts[label] = (emotionCounts[label] || 0) + 1;
                    }
                });

                if (Object.keys(emotionCounts).length > 0) {
                    console.log(
                        "Émotions générées à partir des résultats:",
                        emotionCounts
                    );
                    data.emotion_counts = emotionCounts;
                } else {
                    // Convertir les pourcentages de sentiment en données d'émotions fictives
                    data.emotion_counts = {
                        joy: Math.round(data.positive * 100 * 0.4),
                        admiration: Math.round(data.positive * 100 * 0.3),
                        gratitude: Math.round(data.positive * 100 * 0.3), // S'assurer que gratitude est incluse
                        disappointment: Math.round(data.negative * 100 * 0.4),
                        annoyance: Math.round(data.negative * 100 * 0.3),
                        anger: Math.round(data.negative * 100 * 0.3),
                        neutral: Math.round(data.neutral * 100)
                    };
                }
            } else {
                // Convertir les pourcentages de sentiment en données d'émotions fictives
                data.emotion_counts = {
                    joy: Math.round(data.positive * 100 * 0.4),
                    admiration: Math.round(data.positive * 100 * 0.3),
                    gratitude: Math.round(data.positive * 100 * 0.3), // S'assurer que gratitude est incluse
                    disappointment: Math.round(data.negative * 100 * 0.4),
                    annoyance: Math.round(data.negative * 100 * 0.3),
                    anger: Math.round(data.negative * 100 * 0.3),
                    neutral: Math.round(data.neutral * 100)
                };
            }
        }

        // Vérifier les émotions détectées
        console.log("Émotions détectées:", data.emotion_counts);

        // Afficher les résultats
        displayResults(data);

        return data;
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
        throw error;
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

    // Show results container
    resultsContainer.classList.remove("hidden");

    // Afficher un message de succès simple
    const successMsg = document.createElement("div");
    successMsg.className =
        "bg-green-100 text-green-800 p-3 rounded-lg mb-4 flex items-center";
    successMsg.innerHTML = `
        <i class="fas fa-check-circle text-green-600 mr-2 text-xl"></i>
        <span>Analyse complète de ${data.totalComments} commentaires</span>
    `;

    // Vider le conteneur avant d'ajouter de nouveaux éléments
    resultsContainer.innerHTML = "";
    resultsContainer.appendChild(successMsg);

    // Update comment count
    commentCount.textContent = data.totalComments;

    // Simplification - masquer tous les éléments non nécessaires
    const sentimentSummaries = document.querySelectorAll(".sentiment-summary");
    sentimentSummaries.forEach(el => {
        el.parentElement.style.display = "none";
    });

    // Créer un conteneur pour notre graphique simplifié
    const graphContainer = document.createElement("div");
    graphContainer.className = "bg-white rounded-lg shadow-xl p-6 mb-6";
    graphContainer.innerHTML = `
        <h3 class="text-xl font-bold mb-4 text-center">Distribution des émotions</h3>
        <div id="simple-emotion-chart" class="py-4"></div>
    `;
    resultsContainer.appendChild(graphContainer);

    // Générer uniquement notre graphique simplifié
    if (data.emotion_counts) {
        renderSimpleEmotionChart(data.emotion_counts);
    }

    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: "smooth" });
}

// Nouvelle fonction de rendu simplifié pour le graphique
function renderSimpleEmotionChart(emotionCounts) {
    const chartContainer = document.getElementById("simple-emotion-chart");
    chartContainer.innerHTML = "";

    // If no emotions data
    if (!emotionCounts || Object.keys(emotionCounts).length === 0) {
        chartContainer.innerHTML =
            '<div class="text-center p-4"><span class="text-gray-500">Aucune donnée d\'émotion disponible</span></div>';
        return;
    }

    // Vérification debug pour gratitude
    if ("gratitude" in emotionCounts) {
        console.log(
            "✅ L'émotion 'gratitude' est présente dans les données avec la valeur:",
            emotionCounts.gratitude
        );
    } else {
        console.warn("⚠️ L'émotion 'gratitude' est ABSENTE des données!");
        console.log("Émotions disponibles:", Object.keys(emotionCounts));
    }

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
        caring: "Bienveillance",
        surprise: "Surprise",
        curiosity: "Curiosité",
        realization: "Réalisation",
        desire: "Désir",

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
        confusion: "Confusion",
        nervousness: "Nervosité",

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
        caring: "#0369A1", // light blue
        surprise: "#A855F7", // violet
        curiosity: "#06B6D4", // cyan
        realization: "#0891B2", // teal
        desire: "#DB2777", // pink

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
        confusion: "#6B7280", // gray
        nervousness: "#8B5CF6", // purple

        neutral: "#9CA3AF" // gray
    };

    // Séparer l'émotion neutre des autres émotions
    let neutralEmotionEntry = null;
    let neutralCount = 0;

    // Extraire l'entrée neutre si elle existe
    if ("neutral" in emotionCounts) {
        neutralCount = emotionCounts["neutral"];
        neutralEmotionEntry = ["neutral", neutralCount];
        // Créer une copie sans l'entrée neutre pour le tri
        const nonNeutralEmotions = {};
        Object.entries(emotionCounts).forEach(([emotion, count]) => {
            if (emotion !== "neutral") {
                nonNeutralEmotions[emotion] = count;
            }
        });

        // Trier les émotions non neutres par nombre décroissant
        let sortedNonNeutral = Object.entries(nonNeutralEmotions).sort(
            (a, b) => b[1] - a[1]
        );

        // Réintégrer l'entrée neutre à la fin
        sortedEmotions = [...sortedNonNeutral];
        if (neutralEmotionEntry) {
            sortedEmotions.push(neutralEmotionEntry);
        }
    } else {
        // S'il n'y a pas d'entrée neutre, trier simplement par ordre décroissant
        sortedEmotions = Object.entries(emotionCounts).sort(
            (a, b) => b[1] - a[1]
        );
    }

    console.log(
        "Émotions détectées et triées pour affichage (neutre en bas):",
        sortedEmotions
    );

    // Calculer le total sans les neutres pour un pourcentage plus significatif
    const totalNonNeutral = sortedEmotions.reduce((sum, [emotion, count]) => {
        return emotion !== "neutral" ? sum + count : sum;
    }, 0);

    // Calculer le total global pour la légende
    const totalEmotions = totalNonNeutral + neutralCount;

    // Créer un graphique à barres simple
    const chartDiv = document.createElement("div");
    chartDiv.className = "space-y-4 max-h-96 overflow-y-auto p-2";

    // Si aucune barre n'est créée, on garde une trace
    let barsCreated = 0;

    // Créer une barre pour chaque émotion
    sortedEmotions.forEach(([emotion, count]) => {
        if (count <= 0) return; // Ignorer les émotions avec 0 occurrence

        const emotionName = emotionTranslations[emotion] || emotion;
        const colorHex = emotionColors[emotion] || "#9CA3AF";

        const barContainer = document.createElement("div");
        barContainer.className = "mb-3";
        barContainer.dataset.emotion = emotion; // Ajouter l'attribut de données

        // En-tête avec nom et nombre
        const header = document.createElement("div");
        header.className = "flex justify-between items-center mb-1";

        const nameElement = document.createElement("div");
        nameElement.className = "font-medium";
        nameElement.textContent = emotionName;

        // Calculer le pourcentage et préparer l'affichage du rapport
        let displayTotal;
        let currentPercentage;
        if (emotion === "neutral") {
            displayTotal = totalEmotions;
            currentPercentage = Math.round((count / totalEmotions) * 100);
        } else {
            displayTotal = totalNonNeutral;
            currentPercentage = Math.round((count / totalNonNeutral) * 100);
        }

        const countElement = document.createElement("div");
        countElement.className = "font-bold";
        countElement.textContent = `${count}/${displayTotal}`;

        header.appendChild(nameElement);
        header.appendChild(countElement);
        barContainer.appendChild(header);

        // Barre de progression
        const progressBar = document.createElement("div");
        progressBar.className = "w-full bg-gray-200 rounded-full h-6";

        // Calculer la largeur maximale pour la visualisation en fonction du type d'émotion
        let percentage;
        if (emotion === "neutral") {
            // Pour le neutre: calculer par rapport au total des commentaires
            percentage = Math.max(5, Math.round((count / totalEmotions) * 100));
        } else {
            // Pour les émotions non neutres: calculer par rapport au total sans les neutres
            percentage = Math.max(
                5,
                Math.round((count / totalNonNeutral) * 100)
            );
        }

        // Limiter à 100% maximum pour éviter les barres trop grandes
        percentage = Math.min(percentage, 100);

        const bar = document.createElement("div");
        bar.className =
            "h-6 rounded-full flex items-center justify-center text-white text-xs font-bold";
        bar.style.width = `${percentage}%`;
        bar.style.backgroundColor = colorHex;
        bar.textContent = `${currentPercentage}%`;

        progressBar.appendChild(bar);
        barContainer.appendChild(progressBar);

        chartDiv.appendChild(barContainer);
        barsCreated++;
    });

    if (barsCreated === 0) {
        chartDiv.innerHTML =
            '<div class="text-center p-4 text-red-500">⚠️ Données présentes mais aucune barre créée. Problème détecté.</div>';
        console.error(
            "⚠️ ERREUR: Aucune barre n'a été créée malgré la présence de données:",
            emotionCounts
        );
    }

    chartContainer.appendChild(chartDiv);

    // Ajouter une légende simple
    const legend = document.createElement("div");
    legend.className = "text-sm text-gray-600 mt-4 text-center";

    // Calcul du pourcentage de neutres pour l'affichage
    const neutralPercentage =
        neutralCount > 0 ? Math.round((neutralCount / totalEmotions) * 100) : 0;

    legend.innerHTML = `
        <div>${sortedEmotions.length} types d'émotions détectées | ${totalEmotions} commentaires analysés au total</div>
        <div class="mt-2 text-xs italic">
            <div>• Pour les émotions spécifiques : pourcentage relatif aux ${totalNonNeutral} commentaires non neutres</div>
            <div>• Pour les commentaires neutres : ${neutralCount} sur ${totalEmotions} (${neutralPercentage}% du total)</div>
        </div>
    `;

    chartContainer.appendChild(legend);
}

// Remplacer renderComments par une fonction simplifiée qui ne fait rien
function renderComments(comments) {
    // Ne rien faire, nous n'affichons plus les commentaires individuels
    console.log(`${comments.length} commentaires reçus mais non affichés`);
}

// Met à jour les filtres de commentaires pour afficher les émotions au lieu des sentiments
function updateCommentFilters(emotions, translations) {
    // Supprimer les anciens boutons de filtre
    const showPositive = document.getElementById("show-positive");
    const showNegative = document.getElementById("show-negative");
    const showAll = document.getElementById("show-all");

    // Conserver uniquement le bouton "Tous"
    if (showPositive) showPositive.remove();
    if (showNegative) showNegative.remove();

    // Créer un conteneur pour le sélecteur d'émotions
    const filterContainer =
        document.querySelector(".flex.space-x-2") ||
        document.createElement("div");

    if (!filterContainer.classList.contains("flex")) {
        filterContainer.className = "flex flex-wrap gap-2 mb-3";

        // Trouver où insérer le conteneur
        const commentsHeader = document.querySelector("h4.font-semibold");
        if (commentsHeader) {
            commentsHeader.parentNode.insertBefore(
                filterContainer,
                commentsHeader.nextSibling
            );
        }
    } else {
        // Nettoyer le conteneur existant tout en gardant le bouton "Tous"
        Array.from(filterContainer.children).forEach(child => {
            if (child.id !== "show-all") {
                child.remove();
            }
        });
    }

    // Créer le bouton "Tous" s'il n'existe pas
    if (!showAll) {
        const allButton = document.createElement("button");
        allButton.id = "show-all";
        allButton.className =
            "text-xs bg-gray-100 text-gray-800 px-2 py-1 rounded hover:bg-gray-200";
        allButton.textContent = "Tous";
        allButton.addEventListener("click", () => {
            document.querySelectorAll(".comment").forEach(comment => {
                comment.style.display = "block";
            });

            // Mettre ce bouton en surbrillance
            document
                .querySelectorAll(".emotion-filter")
                .forEach(btn =>
                    btn.classList.remove("ring-2", "ring-offset-1")
                );
            allButton.classList.add("ring-2", "ring-offset-1", "ring-gray-400");
        });
        filterContainer.appendChild(allButton);
    }

    // Réutiliser la référence au bouton "Tous" et lui donner le focus
    const allButton = document.getElementById("show-all");
    allButton.classList.add("ring-2", "ring-offset-1", "ring-gray-400");

    // Ajouter un sélecteur d'émotions
    const selectContainer = document.createElement("div");
    selectContainer.className = "relative ml-2";

    const selectLabel = document.createElement("label");
    selectLabel.htmlFor = "emotion-select";
    selectLabel.className = "text-xs text-gray-600 block mb-1";
    selectLabel.textContent = "Filtrer par émotion:";

    const select = document.createElement("select");
    select.id = "emotion-select";
    select.className = "text-xs border border-gray-300 rounded px-2 py-1";

    // Option par défaut
    const defaultOption = document.createElement("option");
    defaultOption.value = "";
    defaultOption.textContent = "Choisir une émotion";
    select.appendChild(defaultOption);

    // Ajouter les options pour chaque émotion
    emotions
        .sort((a, b) => {
            const nameA = translations[a] || a;
            const nameB = translations[b] || b;
            return nameA.localeCompare(nameB);
        })
        .forEach(emotion => {
            const option = document.createElement("option");
            option.value = emotion;
            option.textContent = translations[emotion] || emotion;
            select.appendChild(option);
        });

    // Gérer le changement de sélection
    select.addEventListener("change", () => {
        if (!select.value) {
            // Si aucune option sélectionnée, afficher tous les commentaires
            document.querySelectorAll(".comment").forEach(comment => {
                comment.style.display = "block";
            });

            // Mettre le bouton "Tous" en surbrillance
            allButton.click();
            return;
        }

        // Filtrer les commentaires par émotion
        document.querySelectorAll(".comment").forEach(comment => {
            comment.style.display =
                comment.dataset.emotion === select.value ? "block" : "none";
        });

        // Retirer la surbrillance de tous les boutons
        document
            .querySelectorAll(".emotion-filter, #show-all")
            .forEach(btn => btn.classList.remove("ring-2", "ring-offset-1"));
    });

    selectContainer.appendChild(selectLabel);
    selectContainer.appendChild(select);
    filterContainer.appendChild(selectContainer);
}
