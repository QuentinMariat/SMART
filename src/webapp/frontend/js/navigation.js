// Navigation module
function initNavigation() {
    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');
    
    // Mobile menu toggle
    mobileMenuButton?.addEventListener('click', () => {
        mobileMenu.classList.toggle('hidden');
    });
    
    // Add click handlers to all nav links
    document.querySelectorAll('.nav-link, .nav-link-mobile, [data-page]').forEach(link => {
        link.addEventListener('click', function() {
            const page = this.dataset.page;
            if (page) {
                showPage(page);
                
                // Update active nav link
                document.querySelectorAll('.nav-link, .nav-link-mobile').forEach(l => {
                    l.classList.remove('active');
                });
                this.classList.add('active');
                
                // Close mobile menu if open
                mobileMenu?.classList.add('hidden');
            }
        });
    });
}

// Show a specific page and hide others
function showPage(pageId) {
    // Hide all pages
    document.querySelectorAll('[id$="-page"]').forEach(page => {
        page.classList.add('hidden');
    });
    
    // Show selected page
    document.getElementById(`${pageId}-page`)?.classList.remove('hidden');
    
    // Scroll to top
    window.scrollTo(0, 0);
}

// Make showPage available globally
window.showPage = showPage;