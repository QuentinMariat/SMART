// Utility functions can be added here
// For example, URL validation functions, etc.

function isValidYoutubeUrl(url) {
    return url.includes('youtube.com/watch');
}

function isValidTwitterUrl(url) {
    return url.includes('twitter.com/');
}

// Export utilities if needed
export { isValidYoutubeUrl, isValidTwitterUrl };