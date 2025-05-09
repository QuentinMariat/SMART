// Analyzer module
function initAnalyzer() {
    // DOM Elements
    const youtubeBtn = document.getElementById('youtube-btn');
    const twitterBtn = document.getElementById('twitter-btn');
    const urlIcon = document.getElementById('url-icon');
    const urlInput = document.getElementById('url-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const sampleBtn = document.getElementById('sample-btn');
    const resultsContainer = document.getElementById('results-container');
    const commentCount = document.getElementById('comment-count');
    const positivePercent = document.getElementById('positive-percent');
    const neutralPercent = document.getElementById('neutral-percent');
    const negativePercent = document.getElementById('negative-percent');
    const positiveBar = document.getElementById('positive-bar');
    const neutralBar = document.getElementById('neutral-bar');
    const negativeBar = document.getElementById('negative-bar');
    const commentsContainer = document.getElementById('comments-container');
    const showPositive = document.getElementById('show-positive');
    const showNegative = document.getElementById('show-negative');
    const showAll = document.getElementById('show-all');
    
    // Sample URLs
    const sampleYoutubeUrl = "https://www.youtube.com/watch?v=dQw4w9WgXcQ";
    const sampleTwitterUrl = "https://twitter.com/elonmusk/status/1234567890";
    
    // Current platform (default: YouTube)
    let currentPlatform = 'youtube';
    
    // Platform selection
    youtubeBtn?.addEventListener('click', () => {
        currentPlatform = 'youtube';
        youtubeBtn.classList.add('active');
        twitterBtn.classList.remove('active');
        urlIcon.className = 'fab fa-youtube platform-icon';
        urlInput.placeholder = 'Paste YouTube video URL here...';
    });
    
    twitterBtn?.addEventListener('click', () => {
        currentPlatform = 'twitter';
        twitterBtn.classList.add('active');
        youtubeBtn.classList.remove('active');
        urlIcon.className = 'fab fa-twitter platform-icon';
        urlInput.placeholder = 'Paste Twitter/X post URL here...';
    });
    
    // Sample URL button
    sampleBtn?.addEventListener('click', () => {
        urlInput.value = currentPlatform === 'youtube' ? sampleYoutubeUrl : sampleTwitterUrl;
    });
    
    // Analyze button
    analyzeBtn?.addEventListener('click', async () => {
        const url = urlInput.value.trim();
        
        if (!url) {
            alert('Please enter a URL to analyze');
            return;
        }
        
        // Validate URL format
        if (currentPlatform === 'youtube' && !url.includes('youtube.com/watch')) {
            alert('Please enter a valid YouTube video URL');
            return;
        }
        
        if (currentPlatform === 'twitter' && !url.includes('twitter.com/')) {
            alert('Please enter a valid Twitter/X post URL');
            return;
        }
        
        // Show loading state
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Analyzing...';
        analyzeBtn.disabled = true;
        
        // Simulate API call with timeout
        setTimeout(() => {
            // For demo purposes, we'll generate mock data
            const mockData = generateMockAnalysis();
            
            // Display results
            displayResults(mockData);
            
            // Reset button
            analyzeBtn.innerHTML = '<i class="fas fa-chart-bar mr-2"></i> Analyze Sentiment';
            analyzeBtn.disabled = false;
        }, 2000);
    });
    
    // Filter comments by sentiment
    showPositive?.addEventListener('click', () => {
        document.querySelectorAll('.comment').forEach(comment => {
            comment.style.display = comment.dataset.sentiment === 'positive' ? 'block' : 'none';
        });
    });
    
    showNegative?.addEventListener('click', () => {
        document.querySelectorAll('.comment').forEach(comment => {
            comment.style.display = comment.dataset.sentiment === 'negative' ? 'block' : 'none';
        });
    });
    
    showAll?.addEventListener('click', () => {
        document.querySelectorAll('.comment').forEach(comment => {
            comment.style.display = 'block';
        });
    });
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
            text: positiveComments[Math.floor(Math.random() * positiveComments.length)],
            sentiment: 'positive',
            likes: Math.floor(Math.random() * 1000),
            time: `${Math.floor(Math.random() * 60)}:${Math.floor(Math.random() * 60).toString().padStart(2, '0')}`
        });
    }
    
    // Generate negative comments
    for (let i = 0; i < Math.floor(negative * 10); i++) {
        comments.push({
            text: negativeComments[Math.floor(Math.random() * negativeComments.length)],
            sentiment: 'negative',
            likes: Math.floor(Math.random() * 500),
            time: `${Math.floor(Math.random() * 60)}:${Math.floor(Math.random() * 60).toString().padStart(2, '0')}`
        });
    }
    
    // Generate neutral comments
    for (let i = 0; i < Math.floor(neutral * 10); i++) {
        comments.push({
            text: neutralComments[Math.floor(Math.random() * neutralComments.length)],
            sentiment: 'neutral',
            likes: Math.floor(Math.random() * 200),
            time: `${Math.floor(Math.random() * 60)}:${Math.floor(Math.random() * 60).toString().padStart(2, '0')}`
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
    const resultsContainer = document.getElementById('results-container');
    const commentCount = document.getElementById('comment-count');
    const positivePercent = document.getElementById('positive-percent');
    const neutralPercent = document.getElementById('neutral-percent');
    const negativePercent = document.getElementById('negative-percent');
    const positiveBar = document.getElementById('positive-bar');
    const neutralBar = document.getElementById('neutral-bar');
    const negativeBar = document.getElementById('negative-bar');
    
    // Show results container
    resultsContainer.classList.remove('hidden');
    
    // Update counts and percentages
    commentCount.textContent = data.totalComments;
    positivePercent.textContent = `${Math.round(data.positive * 100)}%`;
    neutralPercent.textContent = `${Math.round(data.neutral * 100)}%`;
    negativePercent.textContent = `${Math.round(data.negative * 100)}%`;
    
    // Update progress bars
    positiveBar.style.width = `${data.positive * 100}%`;
    neutralBar.style.width = `${data.neutral * 100}%`;
    negativeBar.style.width = `${data.negative * 100}%`;
    
    // Display comments
    renderComments(data.comments);
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
}

// Render comments
function renderComments(comments) {
    const commentsContainer = document.getElementById('comments-container');
    commentsContainer.innerHTML = '';
    
    comments.forEach(comment => {
        const commentDiv = document.createElement('div');
        commentDiv.className = `comment p-3 rounded-lg border ${
            comment.sentiment === 'positive' ? 'border-green-200 bg-green-50' : 
            comment.sentiment === 'negative' ? 'border-red-200 bg-red-50' : 
            'border-gray-200 bg-gray-50'
        }`;
        commentDiv.dataset.sentiment = comment.sentiment;
        
        commentDiv.innerHTML = `
            <div class="flex justify-between items-start mb-1">
                <p class="text-sm font-medium ${
                    comment.sentiment === 'positive' ? 'text-green-800' : 
                    comment.sentiment === 'negative' ? 'text-red-800' : 
                    'text-gray-800'
                }">${comment.text}</p>
                <span class="text-xs text-gray-500 ml-2">${comment.time}</span>
            </div>
            <div class="flex justify-between items-center">
                <span class="text-xs ${
                    comment.sentiment === 'positive' ? 'text-green-600' : 
                    comment.sentiment === 'negative' ? 'text-red-600' : 
                    'text-gray-600'
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