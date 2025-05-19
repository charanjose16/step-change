import { useState, useEffect } from 'react';

/**
 * Custom hook to load requirement data from localStorage.
 * @param {boolean} isOpen - Whether the container (e.g., modal) is open. Triggers loading/reset.
 * @returns {{loadedRequirements: {files: Array<object>} | null, loadError: string, isLoading: boolean}}
 */
export function useRequirementLoader(isOpen) {
    const [loadedRequirements, setLoadedRequirements] = useState(null);
    const [loadError, setLoadError] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        if (isOpen) {
            setIsLoading(true);
            setLoadError("");
            setLoadedRequirements(null);

            // Simulate loading delay for better UX if needed, otherwise remove setTimeout
            const timer = setTimeout(() => {
                const storedResult = localStorage.getItem("requirementsOutput");
                if (storedResult) {
                    try {
                        const parsedResult = JSON.parse(storedResult);
                        if (parsedResult && Array.isArray(parsedResult.requirements)) {
                            setLoadedRequirements({ files: parsedResult.requirements });
                        } else {
                            console.error("Stored data format invalid:", parsedResult);
                            setLoadError("Failed to load results: Invalid data format.");
                            localStorage.removeItem("requirementsOutput"); // Clean up invalid data
                        }
                    } catch (e) {
                        console.error("Failed to parse stored results:", e);
                        setLoadError("Failed to load results: Could not parse data.");
                        localStorage.removeItem("requirementsOutput"); // Clean up corrupted data
                    }
                } else {
                    setLoadError("No analysis results found in storage.");
                }
                setIsLoading(false);
            }, 150); // Optional small delay

            return () => clearTimeout(timer); // Cleanup timer on unmount or if isOpen changes quickly

        } else {
            // Reset state when the modal is closed or hook is inactive
            setIsLoading(false);
            setLoadError("");
            setLoadedRequirements(null);
        }
    }, [isOpen]);

    return { loadedRequirements, loadError, isLoading };
}