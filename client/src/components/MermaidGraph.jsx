import React, { useEffect, useRef, useState, useCallback } from "react";
import mermaid from "mermaid";
import { AlertTriangle, Loader2 } from "lucide-react";

// Enhanced Mermaid Configuration
mermaid.initialize({
  startOnLoad: false,
  theme: "default",
  logLevel: 1,
  securityLevel: "loose",
  htmlLabels: false, // Disable HTML labels for reliable rendering
  fontFamily: "Arial, Helvetica, sans-serif",
  flowchart: {
    useMaxWidth: true,
    htmlLabels: false,
    nodeSpacing: 50,
    rankSpacing: 50,
  },
  er: {
    useMaxWidth: true,
  },
  requirement: {
    useMaxWidth: true,
  },
});

function MermaidGraph({ chart, diagramType, onRenderComplete }) {
  const containerRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [svgContent, setSvgContent] = useState("");

  const validateMermaidSyntax = useCallback((chartText, expectedType) => {
    const validGraphTypes = {
      graph: [/^graph\s+(TD|LR)/i, /^flowchart\s+(TD|LR)/i],
      sequenceDiagram: /^sequenceDiagram/i,
      classDiagram: /^classDiagram/i,
      stateDiagram: /^stateDiagram/i,
      erDiagram: /^erDiagram/i,
      requirementDiagram: /^requirementDiagram/i,
    };

    const cleanChart = chartText.trim();
    const isValidType = Object.entries(validGraphTypes).some(([type, regexes]) => {
      if (Array.isArray(regexes)) {
        return regexes.some((regex) => regex.test(cleanChart)) &&
          (!expectedType || expectedType.toLowerCase().includes(type));
      }
      return regexes.test(cleanChart) &&
        (!expectedType || expectedType.toLowerCase().includes(type));
    });

    const hasValidNodes =
      /\[.*?\]/.test(cleanChart) ||
      /\{.*?\}/.test(cleanChart) ||
      cleanChart.includes("requirement") ||
      /".*?"/.test(cleanChart);

    const hasValidConnections = /-->|---|--/.test(cleanChart);

    let typeMismatch = false;
    if (expectedType) {
      const expectedLower = expectedType.toLowerCase();
      if (expectedLower.includes("entity relationship") && !/^erDiagram/i.test(cleanChart)) {
        typeMismatch = true;
      } else if (expectedLower.includes("requirement") && !/^requirementDiagram/i.test(cleanChart)) {
        typeMismatch = true;
      }
    }

    return { isValid: isValidType && (hasValidNodes || hasValidConnections), typeMismatch };
  }, []);

  useEffect(() => {
    const renderMermaid = async () => {
      try {
        setIsLoading(true);
        setError(null);

        const cleanChart = chart?.trim();
        if (!cleanChart) throw new Error("Empty diagram definition");

        const { isValid, typeMismatch } = validateMermaidSyntax(cleanChart, diagramType);
        if (!isValid) {
          throw new Error("Invalid Mermaid syntax or missing nodes/edges.");
        }
        if (typeMismatch) {
          console.warn("Diagram type mismatch:", {
            expected: diagramType,
            actual: cleanChart.split("\n")[0].trim(),
          });
        }

        const uniqueId = `mermaid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        let { svg } = await mermaid.render(uniqueId, cleanChart);

        if (!svg || svg.trim().length === 0 || !svg.includes("<text")) {
          throw new Error("SVG generated but no visible content found in diagram.");
        }

        // Add explicit width and height to SVG to avoid html2canvas issues
        const width = containerRef.current?.offsetWidth || 800;
        const height = containerRef.current?.offsetHeight || 600;

        // Replace <svg ...> to inject width, height, font and fill styles
        svg = svg.replace(
          /^<svg([^>]*)>/,
          `<svg$1 width="${width}" height="${height}" style="font-family: Arial, sans-serif; font-size: 14px;">`
        );

        // Embed CSS styles for visible text and labels (dark fill)
        svg = svg.replace(
          "</svg>",
          `<style>
            text, .nodeLabel, .edgeLabel, .label {
              font-family: Arial, sans-serif !important;
              font-size: 14px !important;
              fill: #222 !important;
            }
          </style></svg>`
        );

        setSvgContent(svg);

        if (onRenderComplete) {
          setTimeout(() => onRenderComplete(), 500);
        }
      } catch (err) {
        setError({
          message: "Diagram Rendering Failed",
          str: err.message || "Unknown rendering error",
          code: chart,
        });
        setSvgContent("");
      } finally {
        setIsLoading(false);
      }
    };

    renderMermaid();
  }, [chart, diagramType, validateMermaidSyntax, onRenderComplete]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full flex items-center justify-center mermaid-container relative"
      style={{ minHeight: "400px" }}
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
          <span className="text-sm font-semibold mb-1">{error.message}</span>
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
          className="w-full h-full flex items-center justify-center overflow-auto"
          dangerouslySetInnerHTML={{ __html: svgContent }}
        />
      )}
    </div>
  );
}

export default MermaidGraph;
