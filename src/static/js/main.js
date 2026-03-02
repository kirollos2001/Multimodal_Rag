/* ═══════════════════════════════════════════════════
   NANO BANANA — Chat UI Logic
   ═══════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {

    // ─── DOM Elements ─────────────────────────────────
    const chatForm = document.getElementById('chatForm');
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const chatMessages = document.getElementById('chatMessages');
    const welcomeScreen = document.getElementById('welcomeScreen');
    const attachBtn = document.getElementById('attachBtn');
    const imageInput = document.getElementById('imageInput');
    const attachedPreview = document.getElementById('attachedPreview');
    const previewThumb = document.getElementById('previewThumb');
    const previewName = document.getElementById('previewName');
    const removeAttachment = document.getElementById('removeAttachment');
    const newChatBtn = document.getElementById('newChatBtn');
    const clearChatBtn = document.getElementById('clearChatBtn');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebar = document.getElementById('sidebar');
    const chatHistory = document.getElementById('chatHistory');

    // ─── State ────────────────────────────────────────
    let selectedFile = null;
    let selectedFileDataURL = null;
    let conversationHistory = [];
    let isProcessing = false;

    // ─── Auto-resize textarea ─────────────────────────
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 150) + 'px';
        updateSendButton();
    });

    function updateSendButton() {
        const hasContent = messageInput.value.trim().length > 0 || selectedFile;
        sendBtn.classList.toggle('active', hasContent && !isProcessing);
    }

    // ─── Sidebar Toggle ───────────────────────────────
    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
        sidebar.classList.toggle('open');
    });

    // ─── Image Attachment ─────────────────────────────
    attachBtn.addEventListener('click', () => imageInput.click());

    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    removeAttachment.addEventListener('click', () => {
        clearAttachment();
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) return;
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            selectedFileDataURL = e.target.result;
            previewThumb.style.backgroundImage = `url(${e.target.result})`;
            previewName.textContent = file.name;
            attachedPreview.classList.remove('hidden');
            updateSendButton();
        };
        reader.readAsDataURL(file);
    }

    function clearAttachment() {
        selectedFile = null;
        selectedFileDataURL = null;
        imageInput.value = '';
        attachedPreview.classList.add('hidden');
        updateSendButton();
    }

    // ─── Suggestion Chips ─────────────────────────────
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const query = chip.dataset.query;
            messageInput.value = query;
            messageInput.style.height = 'auto';
            messageInput.style.height = messageInput.scrollHeight + 'px';
            updateSendButton();
            messageInput.focus();
        });
    });

    // ─── New Chat / Clear ─────────────────────────────
    newChatBtn.addEventListener('click', startNewChat);
    clearChatBtn.addEventListener('click', startNewChat);

    function startNewChat() {
        conversationHistory = [];
        chatMessages.innerHTML = '';
        chatMessages.appendChild(createWelcomeScreen());
        lucide.createIcons();
        messageInput.value = '';
        messageInput.style.height = 'auto';
        clearAttachment();
        updateSendButton();
    }

    function createWelcomeScreen() {
        const div = document.createElement('div');
        div.className = 'welcome-screen';
        div.id = 'welcomeScreen';
        div.innerHTML = `
            <div class="welcome-logo">
                <div class="welcome-icon-ring">
                    <i data-lucide="sparkles"></i>
                </div>
            </div>
            <h1 class="welcome-title">أهلاً! أنا مساعدك الذكي 👋</h1>
            <p class="welcome-subtitle">ابعتلي صورة أو اكتبلي اللي بتدور عليه وأنا أجيبهولك</p>
            <div class="suggestion-chips">
                <button class="chip" data-query="عايز جاكت شتوي أسود">
                    <i data-lucide="shirt"></i>
                    <span>جاكت شتوي أسود</span>
                </button>
                <button class="chip" data-query="Show me casual outfits under 1000 EGP">
                    <i data-lucide="tag"></i>
                    <span>Casual under 1000</span>
                </button>
                <button class="chip" data-query="هودي oversize">
                    <i data-lucide="zap"></i>
                    <span>هودي Oversize</span>
                </button>
                <button class="chip" data-query="What styles do you have?">
                    <i data-lucide="palette"></i>
                    <span>What styles?</span>
                </button>
            </div>
        `;
        // Re-attach chip listeners
        div.querySelectorAll('.chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const query = chip.dataset.query;
                messageInput.value = query;
                messageInput.style.height = 'auto';
                messageInput.style.height = messageInput.scrollHeight + 'px';
                updateSendButton();
                messageInput.focus();
            });
        });
        return div;
    }

    // ─── Form Submit (Send Message) ───────────────────
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (isProcessing) return;

        const text = messageInput.value.trim();
        if (!text && !selectedFile) return;

        // Remove welcome screen
        const ws = document.getElementById('welcomeScreen');
        if (ws) ws.remove();

        // Add user message bubble
        addUserMessage(text, selectedFileDataURL);

        // Save to history
        conversationHistory.push({ role: 'user', content: text });

        // Reset input
        const currentFile = selectedFile;
        messageInput.value = '';
        messageInput.style.height = 'auto';

        // Show typing indicator
        isProcessing = true;
        updateSendButton();
        const typingEl = addTypingIndicator();

        // Build form data
        const formData = new FormData();
        if (text) formData.append('message', text);
        if (currentFile) formData.append('image_file', currentFile);
        formData.append('conversation_history', JSON.stringify(conversationHistory));

        clearAttachment();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();

            // Remove typing indicator
            typingEl.remove();

            // Add AI message
            addAssistantMessage(data.reply, data.products);

            // Save to history
            conversationHistory.push({ role: 'assistant', content: data.reply });

            // Update sidebar history
            updateSidebarHistory(text);

        } catch (error) {
            console.error('Chat error:', error);
            typingEl.remove();
            addAssistantMessage('عذرًا، حصل مشكلة في الاتصال. حاول تاني. 🔄');
        } finally {
            isProcessing = false;
            updateSendButton();
        }
    });

    // Enter to send (Shift+Enter for newline)
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    // ─── Message Rendering ────────────────────────────

    function addUserMessage(text, imageDataURL) {
        const row = document.createElement('div');
        row.className = 'message-row user';

        let imageHTML = '';
        if (imageDataURL) {
            imageHTML = `<img src="${imageDataURL}" alt="Attached image" class="user-image-attachment">`;
        }

        row.innerHTML = `
            <div class="message-bubble">${imageHTML}${escapeHtml(text)}</div>
        `;
        chatMessages.appendChild(row);
        scrollToBottom();
    }

    function addAssistantMessage(text, products) {
        const row = document.createElement('div');
        row.className = 'message-row assistant';

        let productsHTML = '';
        if (products && products.length > 0) {
            productsHTML = `<div class="product-list">${products.map(p => {
                // Build images array from the new `images` field, fallback to old `image` field
                let imageFiles = [];
                if (p.images && p.images.length > 0) {
                    imageFiles = p.images;
                } else if (p.image) {
                    imageFiles = [p.image];
                }

                const imagesHTML = imageFiles.length > 0
                    ? imageFiles.map(img => `
                        <img src="/images/${p.folder}/${img}" alt="Product" 
                             onerror="this.src='https://placehold.co/300x400?text=No+Image'">
                    `).join('')
                    : `<img src="https://placehold.co/300x400?text=No+Image" alt="No Image">`;

                const price = p.price ? `${p.price} EGP` : '';
                const score = p.score ? `${(p.score * 100).toFixed(0)}% match` : '';
                const desc = p.description || '';
                const idText = p.id || p.folder || 'N/A';
                const color = p.color || '';
                const category = p.category || '';

                return `
                    <div class="product-item">
                        <div class="product-item-images">
                            ${imagesHTML}
                        </div>
                        <div class="product-item-details">
                            <div class="product-item-header">
                                <span class="product-item-id">ID: ${idText}</span>
                                ${score ? `<span class="product-item-score">${score}</span>` : ''}
                            </div>
                            <div class="product-item-tags">
                                ${price ? `<span class="product-item-price">${price}</span>` : ''}
                                ${color ? `<span class="product-item-tag">${color}</span>` : ''}
                                ${category ? `<span class="product-item-tag">${category}</span>` : ''}
                            </div>
                            ${desc ? `<div class="product-item-desc">${desc}</div>` : ''}
                        </div>
                    </div>
                `;
            }).join('')}</div>`;
        }

        // Format the AI text (basic markdown-like: **bold**, newlines)
        const formattedText = formatAIText(text);

        row.innerHTML = `
            <div class="message-avatar">
                <i data-lucide="sparkles"></i>
            </div>
            <div class="message-bubble">${formattedText}${productsHTML}</div>
        `;
        chatMessages.appendChild(row);
        lucide.createIcons();
        scrollToBottom();
    }

    function addTypingIndicator() {
        const row = document.createElement('div');
        row.className = 'message-row assistant';
        row.id = 'typingRow';
        row.innerHTML = `
            <div class="message-avatar">
                <i data-lucide="sparkles"></i>
            </div>
            <div class="message-bubble"><div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div></div>
        `;
        chatMessages.appendChild(row);
        lucide.createIcons();
        scrollToBottom();
        return row;
    }

    // ─── Utilities ────────────────────────────────────

    function scrollToBottom() {
        requestAnimationFrame(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function formatAIText(text) {
        if (!text) return '';
        // Escape HTML first
        let safe = escapeHtml(text);
        // Bold: **text**
        safe = safe.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Newlines
        safe = safe.replace(/\n/g, '<br>');
        return safe;
    }

    function updateSidebarHistory(text) {
        // Add to sidebar if it's real user text
        if (!text || text.length < 2) return;
        const label = text.length > 30 ? text.substring(0, 30) + '…' : text;

        const item = document.createElement('div');
        item.className = 'history-item';
        item.textContent = label;
        item.title = text;

        // Insert at top
        if (chatHistory.firstChild) {
            chatHistory.insertBefore(item, chatHistory.firstChild);
        } else {
            chatHistory.appendChild(item);
        }

        // Limit to 15 items
        while (chatHistory.children.length > 15) {
            chatHistory.removeChild(chatHistory.lastChild);
        }
    }

    // ─── Drag and Drop on entire chat area ────────────
    const chatMain = document.querySelector('.chat-main');

    chatMain.addEventListener('dragover', (e) => {
        e.preventDefault();
        chatMain.style.outline = '2px dashed var(--accent)';
        chatMain.style.outlineOffset = '-4px';
    });

    chatMain.addEventListener('dragleave', (e) => {
        e.preventDefault();
        chatMain.style.outline = 'none';
    });

    chatMain.addEventListener('drop', (e) => {
        e.preventDefault();
        chatMain.style.outline = 'none';
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

});
