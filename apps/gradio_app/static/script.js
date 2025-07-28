function copyToClipboard(elementId) {
    const element = document.getElementById(elementId);
    let text = element.innerText.replace(/^Thinking Process:\\n|^Final Answer:\\n/, '');
    text = text.replace(/\\mjx-[^\\s]+/g, '');
    navigator.clipboard.writeText(text).then(() => {
        alert('Copied to clipboard!');
    }).catch(err => {
        console.error('Failed to copy: ', err);
    });
}