document.addEventListener('DOMContentLoaded', function() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const emailText = document.getElementById('emailText');
    const modelSelect = document.getElementById('modelType');
    const verdictDiv = document.getElementById('verdict');
    const highlightedTextDiv = document.getElementById('highlightedText');
    const limeToggle = document.getElementById('limeToggle');
    const loadingOverlay = document.getElementById('loadingOverlay');
    
    // Check if all required elements exist
    if (!analyzeBtn || !emailText || !modelSelect || !verdictDiv || !highlightedTextDiv || !limeToggle || !loadingOverlay) {
        console.error('Required elements not found:', {
            analyzeBtn: !!analyzeBtn,
            emailText: !!emailText,
            modelSelect: !!modelSelect,
            verdictDiv: !!verdictDiv,
            highlightedTextDiv: !!highlightedTextDiv,
            limeToggle: !!limeToggle,
            loadingOverlay: !!loadingOverlay
        });
        return;
    }
    
    // Initialize the chart with dark mode colors
    const ctx = document.getElementById('confidenceChart').getContext('2d');
    const confidenceChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Legitimate', 'Phishing'],
            datasets: [{
                data: [50, 50],
                backgroundColor: [
                    '#28a745',  // Green for legitimate
                    '#dc3545'   // Red for phishing
                ],
                borderColor: [
                    '#1e7e34',  // Darker green
                    '#bd2130'   // Darker red
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#e0e0e0',  // Light text for dark mode
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.raw}%`;
                        }
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            }
        }
    });

    console.log('Page loaded, elements found:', {
        analyzeBtn: !!analyzeBtn,
        emailText: !!emailText,
        modelSelect: !!modelSelect,
        verdictDiv: !!verdictDiv,
        highlightedTextDiv: !!highlightedTextDiv,
        limeToggle: !!limeToggle,
        loadingOverlay: !!loadingOverlay
    });

    analyzeBtn.addEventListener('click', async function() {
        console.log('Analyze button clicked');
        const text = emailText.value.trim();
        if (!text) {
            alert('Please enter some email text to analyze.');
            return;
        }

        // Show loading state
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = 'Analyzing...';
        loadingOverlay.classList.add('active');

        try {
            console.log('Sending request to server...');
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    model_type: modelSelect.value,
                    show_lime: limeToggle.checked
                })
            });

            console.log('Response received:', response.status);
            const data = await response.json();
            console.log('Data received:', data);
            
            // Update verdict
            verdictDiv.className = 'alert ' + (data.is_phishing ? 'phishing' : 'legitimate');
            verdictDiv.textContent = data.is_phishing ? '⚠️ This is a phishing email!' : '✅ This appears to be a legitimate email.';
            
            // Update chart
            if (data.confidence !== null) {
                // Full mode: show actual confidence
                const confidence = data.confidence * 100;
                verdictDiv.textContent += ` (Confidence: ${confidence.toFixed(1)}%)`;
                
                if (data.is_phishing) {
                    confidenceChart.data.datasets[0].data = [100 - confidence, confidence];
                } else {
                    confidenceChart.data.datasets[0].data = [confidence, 100 - confidence];
                }
            } else {
                // Chunk mode: show neutral state with a slight bias based on the result
                if (data.is_phishing) {
                    confidenceChart.data.datasets[0].data = [40, 60];
                } else {
                    confidenceChart.data.datasets[0].data = [60, 40];
                }
                verdictDiv.textContent += ' (Chunk Mode)';
            }
            confidenceChart.update();

            // Update highlighted text
            let highlightedText = text;
            
            // Add LIME explanations (only if enabled)
            if (limeToggle.checked && data.lime_explanation) {
                // First, split the text into sentences
                const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
                
                // Create a map of sentences to their LIME words
                const sentenceHighlights = new Map();
                data.lime_explanation.forEach(item => {
                    if (item.is_positive) {  // Only display positive indicators
                        const regex = new RegExp(`\\b${item.word}\\b`, 'gi');
                        sentences.forEach(sentence => {
                            if (regex.test(sentence)) {
                                if (!sentenceHighlights.has(sentence)) {
                                    sentenceHighlights.set(sentence, []);
                                }
                                sentenceHighlights.get(sentence).push(item);
                            }
                        });
                    }
                });
                
                // Highlight sentences that contain LIME words
                sentenceHighlights.forEach((items, sentence) => {
                    const maxWeight = Math.max(...items.map(item => item.weight));
                    const title = `LIME (${(maxWeight * 100).toFixed(1)}%)`;
                    highlightedText = highlightedText.replace(sentence, match => 
                        `<span class="highlighted-text" style="background-color: #6f42c1;" title="${title}">${match}</span>`
                    );
                });
            }
            
            // Add urgency explanations (keep as individual words)
            if (data.urgency_explanation) {
                data.urgency_explanation.forEach(item => {
                    const regex = new RegExp(`\\b${item.word}\\b`, 'gi');
                    highlightedText = highlightedText.replace(regex, match => 
                        `<span class="highlighted-text" style="background-color: #ffc107;" title="URGENCY (${(item.weight * 100).toFixed(1)}%)">${match}</span>`
                    );
                });
            }

            highlightedTextDiv.innerHTML = highlightedText;
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while analyzing the email. Please try again.');
        } finally {
            // Reset button state and hide loading overlay
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = 'Analyze Email';
            loadingOverlay.classList.remove('active');
        }
    });
});

function updateHighlightedText(text, highlights) {
    const container = document.getElementById('highlighted-container');
    if (!container) return;

    // Clear existing content
    container.innerHTML = '';

    if (!text || !highlights) {
        container.textContent = text || '';
        return;
    }

    // Sort highlights by start position
    highlights.sort((a, b) => a.start - b.start);

    let lastIndex = 0;
    const fragment = document.createDocumentFragment();

    highlights.forEach(highlight => {
        // Add text before highlight
        if (highlight.start > lastIndex) {
            const textBefore = document.createTextNode(text.slice(lastIndex, highlight.start));
            fragment.appendChild(textBefore);
        }

        // Add highlighted text
        const highlightedSpan = document.createElement('span');
        highlightedSpan.className = 'highlighted-text';
        highlightedSpan.style.backgroundColor = highlight.color;
        highlightedSpan.textContent = text.slice(highlight.start, highlight.end);
        fragment.appendChild(highlightedSpan);

        lastIndex = highlight.end;
    });

    // Add remaining text
    if (lastIndex < text.length) {
        const remainingText = document.createTextNode(text.slice(lastIndex));
        fragment.appendChild(remainingText);
    }

    container.appendChild(fragment);
} 