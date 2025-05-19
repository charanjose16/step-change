/**
 * Trims a string to a maximum length, adding an ellipsis if truncated.
 * @param {string | null | undefined} text - The text to trim.
 * @param {number} [max=150] - The maximum length before trimming.
 * @returns {string} The trimmed text or a default message.
 */
export const trimDescription = (text, max = 150) => {
    if (!text) return "No description available.";
    return text.length <= max ? text : text.substring(0, max) + "...";
};