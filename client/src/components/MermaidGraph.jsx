import React, { useEffect, useRef, useState } from "react";
import mermaid from "mermaid";
import { AlertTriangle, Loader2 } from "lucide-react";

// Initialize Mermaid (ensure this is outside the component)
mermaid.initialize({
    startOnLoad: false,
    theme: "base",
    // securityLevel: 'loose', // Consider if needed
});

function MermaidGraph({ chart }) {
    const containerRef = useRef(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null); // Stores { message, str, code }
    const [svgContent, setSvgContent] = useState("");
    // Keep track of the chart being processed to avoid race conditions
    const renderingChartRef = useRef(null);

    useEffect(() => {
        // Store the chart value this effect is processing
        renderingChartRef.current = chart;

        // Reset state for the new chart
        setIsLoading(true);
        setError(null);
        setSvgContent(""); // Clear previous SVG

        // --- Validation ---
        if (
            !chart ||
            typeof chart !== "string" ||
            !chart
                .trim()
                .match(
                    /^(graph|flowchart|sequenceDiagram|gantt|classDiagram|stateDiagram|pie|erDiagram|journey|requirementDiagram)/i
                )
        ) {
            setError({
                message: "Invalid Graph Definition",
                str: "The provided text doesn't look like a valid Mermaid diagram type or is empty.",
                code: chart || ""
            });
            setIsLoading(false);
            return; // Stop processing if invalid
        }

        // --- Rendering ---
        // Ensure the container is available (it should be, as it's always rendered now)
        if (!containerRef.current) {
            // This should ideally not happen with the new structure, but acts as a safeguard
            console.error("MermaidGraph: containerRef is null during rendering attempt.");
            setError({ message: "Internal Error: Container not ready." });
            setIsLoading(false);
            return;
        }

        const renderMermaid = async () => {
            try {
                const uniqueId = `mermaid-graph-${Date.now()}-${Math.random().toString(16).substring(2, 8)}`;
                const { svg } = await mermaid.render(uniqueId, chart);

                // Check if the chart prop has changed *since* this render started
                if (renderingChartRef.current === chart) {
                    setSvgContent(svg);
                    setError(null); // Clear previous error on success
                } else {
                     console.log("MermaidGraph: Stale render ignored.");
                }

            } catch (err) {
                 // Check if the chart prop has changed *since* this render started
                if (renderingChartRef.current === chart) {
                    console.error("Mermaid rendering error:", err);
                    setError({
                        message: err.message || "Unknown Mermaid rendering error.",
                        str: err.str || "Syntax error in diagram definition.",
                        code: chart
                    });
                    setSvgContent(""); // Clear any potentially broken SVG content
                } else {
                     console.log("MermaidGraph: Stale error ignored.");
                }
            } finally {
                 // Only stop loading if this is the effect for the *current* chart
                 if (renderingChartRef.current === chart) {
                    setIsLoading(false);
                 }
            }
        };

        renderMermaid();

    }, [chart]); // Re-run effect if chart content changes


    // --- RENDERING LOGIC ---
    // Always render the container div with the ref
    return (
        <div
            ref={containerRef}
            className="w-full h-full flex items-center justify-center mermaid-container relative" // Added relative for potential absolute positioning of overlays
        >
            {/* Loading Overlay */}
            {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75 z-10">
                     <div className="flex items-center text-slate-500 p-4">
                        <Loader2 className="w-6 h-6 mr-2 animate-spin" />
                        Rendering graph...
                    </div>
                </div>
            )}

            {/* Error Display */}
            {!isLoading && error && (
                 <div className="flex flex-col items-center justify-center h-full text-red-600 p-4 text-center w-full bg-red-50 rounded border border-red-200">
                    <AlertTriangle className="w-8 h-8 mb-2 flex-shrink-0" />
                    <span className="text-sm font-semibold mb-1">
                        {error.message === "Invalid Graph Definition"
                            ? "Invalid Graph Definition"
                            : error.message?.includes("Container not ready")
                            ? "Rendering Error"
                            : "Mermaid Syntax Error"}
                    </span>
                    <p className="text-xs mb-2">{error.str || error.message || "An unexpected error occurred."}</p>
                    {error.code && error.message !== "Invalid Graph Definition" && !error.message?.includes("Container not ready") && (
                        <details className="w-full text-left mt-2 text-xs text-slate-700 bg-white p-2 border rounded max-h-40 overflow-auto">
                            <summary className="cursor-pointer font-medium text-slate-600 hover:text-slate-800">Show problematic code</summary>
                            <pre className="mt-1 whitespace-pre-wrap break-all">
                                <code>{error.code}</code>
                            </pre>
                        </details>
                    )}
                </div>
            )}

            {/* SVG Content (only rendered when not loading and no error) */}
            {!isLoading && !error && (
                <div
                    className="w-full h-full flex items-center justify-center" // Inner div to contain the SVG
                    dangerouslySetInnerHTML={{ __html: svgContent }}
                />
            )}
        </div>
    );
}

export default MermaidGraph;