import React, { useEffect, useRef, useState, useCallback } from "react";
import mermaid from "mermaid";
import { AlertTriangle, Loader2 } from "lucide-react";

// Advanced Mermaid Configuration
mermaid.initialize({
    startOnLoad: false,
    theme: "default",  // Options: default, dark, forest
    logLevel: 4,  // Verbose logging
    securityLevel: 'strict',
    maxTextSize: 100000,  // Adjust based on your needs
});

function MermaidGraph({ chart }) {
    const containerRef = useRef(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const [svgContent, setSvgContent] = useState("");

    // Strict Mermaid Syntax Validation
    const validateMermaidSyntax = useCallback((chartText) => {
        const validGraphTypes = [
            /^graph\s+(TD|LR)/i,
            /^flowchart\s+(TD|LR)/i
        ];

        const isValidType = validGraphTypes.some(regex => regex.test(chartText.trim()));
        const hasValidNodes = /\[.*?\]/.test(chartText);
        const hasValidConnections = /-->|---/.test(chartText);

        return isValidType && hasValidNodes && hasValidConnections;
    }, []);

    useEffect(() => {
        const renderMermaid = async () => {
            try {
                setIsLoading(true);
                setError(null);

                // Trim and validate chart
                const cleanChart = chart.trim();
                
                if (!cleanChart) {
                    throw new Error("Empty diagram definition");
                }

                if (!validateMermaidSyntax(cleanChart)) {
                    throw new Error("Invalid Mermaid syntax. Must use graph TD/LR with [nodes] and --> connections");
                }

                // Generate unique ID to prevent conflicts
                const uniqueId = `mermaid-${Date.now()}`;
                
                // Render Mermaid diagram
                const { svg } = await mermaid.render(uniqueId, cleanChart);

                if (!svg || svg.trim().length === 0) {
                    throw new Error("Generated SVG is empty");
                }

                setSvgContent(svg);
                setError(null);
            } catch (err) {
                console.error("Mermaid Rendering Error:", err);
                setError({
                    message: "Diagram Rendering Failed",
                    str: err.message || "Unable to generate diagram",
                    code: chart
                });
                setSvgContent("");
            } finally {
                setIsLoading(false);
            }
        };

        renderMermaid();
    }, [chart, validateMermaidSyntax]);

    return (
        <div 
            ref={containerRef}
            className="w-full h-full flex items-center justify-center mermaid-container relative"
        >
            {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75 z-10">
                    <div className="flex items-center text-slate-500 p-4">
                        <Loader2 className="w-6 h-6 mr-2 animate-spin" />
                        Rendering graph...
                    </div>
                </div>
            )}

            {!isLoading && error && (
                <div className="flex flex-col items-center justify-center h-full text-red-600 p-4 text-center w-full bg-red-50 rounded border border-red-200">
                    <AlertTriangle className="w-8 h-8 mb-2 flex-shrink-0" />
                    <span className="text-sm font-semibold mb-1">
                        {error.message}
                    </span>
                    <p className="text-xs mb-2">{error.str}</p>
                    {error.code && (
                        <details className="w-full text-left mt-2 text-xs text-slate-700 bg-white p-2 border rounded max-h-40 overflow-auto">
                            <summary className="cursor-pointer font-medium text-slate-600 hover:text-slate-800">
                                Show problematic code
                            </summary>
                            <pre className="mt-1 whitespace-pre-wrap break-all">
                                <code>{error.code}</code>
                            </pre>
                        </details>
                    )}
                </div>
            )}

            {!isLoading && !error && (
                <div
                    className="w-full h-full flex items-center justify-center"
                    dangerouslySetInnerHTML={{ __html: svgContent }}
                />
            )}
        </div>
    );
}

export default MermaidGraph;
