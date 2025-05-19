// Helper function to validate GitHub URL
const isValidGithubUrl = (url) => {
    try {
        const parsed = new URL(url);
        return parsed.hostname.toLowerCase() === "github.com";
    } catch {
        return false;
    }
};

// Helper function to validate Windows absolute file paths
const isValidFilePath = (path) => {
    // Windows absolute path: e.g., C:\path\to\folder or C:\path\to\file
    const windowsRegex = /^[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*$/;
    return windowsRegex.test(path);
};

export { isValidGithubUrl, isValidFilePath };