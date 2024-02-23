function sendMessage() {
    var input = document.getElementById("user-input");
    var message = input.value.trim();
    if (message !== "") {
        displayMessage("user", message);
        // 简单的响应逻辑
        displayMessage("bot", "这是一个自动回复: " + message);
    }
    input.value = ""; // 清空输入框
}

function displayMessage(sender, message) {
    var chatWindow = document.getElementById("chat-window");
    var msgDiv = document.createElement("div");
    msgDiv.textContent = sender + ": " + message;
    chatWindow.appendChild(msgDiv);
}
