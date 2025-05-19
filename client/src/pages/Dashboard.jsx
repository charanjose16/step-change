import React, { useState, useEffect, Suspense, lazy } from "react";
import axios from "../api/axiosConfig";
import Navbar from "../components/Navbar";
import { XCircle, CheckCircle, AlertCircle, Download, RotateCcw } from "lucide-react"; // Added Download & RotateCcw icons
import { isValidGithubUrl, isValidFilePath } from "../utils/validation";
import { jsPDF } from "jspdf";

const ResultViewer = lazy(() => import("../components/ResultViewer"));

export default function Dashboard() {
    const [link, setLink] = useState("");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [showResultsViewer, setShowResultsViewer] = useState(false);
    const [analysisComplete, setAnalysisComplete] = useState(false); // Track analysis success

    // Load previous results from local storage on mount
    useEffect(() => {
        const storedResult = localStorage.getItem("requirementsOutput");
        if (storedResult) {
            try {
                const parsedResult = JSON.parse(storedResult);
                setResult(parsedResult);
                setAnalysisComplete(true); // Set analysis complete if results are loaded
                // --- Add this line to automatically show the viewer ---
                
                // ------------------------------------------------------
            } catch (e) {
                console.error("Failed to parse stored results:", e);
                localStorage.removeItem("requirementsOutput"); // Clear invalid data
            }
        }
    }, []);  // Empty dependency array ensures this runs only once on mount

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(null);
        setResult(null); // Clear previous results before new analysis
        setShowResultsViewer(false);
        setAnalysisComplete(false); // Reset analysis status
        localStorage.removeItem("requirementsOutput"); // Clear old results from storage

        // Validate the entered link
        if (!isValidGithubUrl(link) && !isValidFilePath(link)) {
            setError("Please enter a valid GitHub URL or an absolute file path.");
            return;
        }

        setLoading(true);

        try {
            // POST to the /analysis endpoint.
            const res = await axios.post("/analysis", { link });
            console.log("Response:", res.data);
            setResult(res.data);
            // Save result to local storage on success
            localStorage.setItem("requirementsOutput", JSON.stringify(res.data));
            setAnalysisComplete(true); // Mark analysis as complete
            setError(null); // Clear any previous errors
        } catch (err) {
            console.error(err);
            const errMsg =
                err.response && err.response.data && err.response.data.detail
                    ? err.response.data.detail
                    : "Failed to analyze codebase.";
            setError(errMsg);
            setResult(null); // Ensure result is null on error
            setAnalysisComplete(false); // Mark analysis as failed/incomplete
            localStorage.removeItem("requirementsOutput"); // Clear storage on error
        } finally {
            setLoading(false);
        }
    };

    // Reset Handler
    const handleReset = () => {
        setResult(null);
        setAnalysisComplete(false);
        setError(null);
        setShowResultsViewer(false); // Close viewer if open
        localStorage.removeItem("requirementsOutput");
        // Optionally clear the input link as well:
        // setLink("");
        console.log("Analysis results cleared.");
    };


    // PDF Download Handler using jsPDF with AutoTable (explicit call)
    const downloadPDF = () => {
        console.log("Result state on download:", JSON.stringify(result, null, 2));

        if (!result || !result.requirements || !Array.isArray(result.requirements) || result.requirements.length === 0) {
            alert("No valid analysis results available to download. Check console for details.");
            console.error("Download PDF failed: Invalid 'result' or 'result.requirements'.", result);
            return;
        }

        const doc = new jsPDF();
        const pageHeight = doc.internal.pageSize.height || doc.internal.pageSize.getHeight();
        const pageWidth = doc.internal.pageSize.width || doc.internal.pageSize.getWidth();
        const margin = 15;
        const maxLineWidth = pageWidth - margin * 2;
        let yPos; // Initialize yPos inside the loop

        // --- PDF Header (Only for the first page initially) ---
        const drawHeader = (docInstance, currentY) => {
            docInstance.setFontSize(18);
            docInstance.setFont(undefined, "bold");
            docInstance.setTextColor(40, 58, 90);
            docInstance.text("Codebase Analysis Report", margin, currentY);
            currentY += 6; // Move down
            docInstance.setDrawColor(40, 58, 90);
            docInstance.line(margin, currentY, pageWidth - margin, currentY); // Underline header
            currentY += 12; // Space after header
            return currentY;
        };

        yPos = margin + 8; // Initial Y position for the first page header
        yPos = drawHeader(doc, yPos);

        // --- Content ---
        result.requirements.forEach((file, index) => {
            // Add a new page for every file *after* the first one
            if (index > 0) {
                doc.addPage();
                yPos = margin + 8; // Reset Y for the new page's header
                // Optionally redraw header on each new page:
                // yPos = drawHeader(doc, yPos);
                // If not redrawing header, just reset yPos for content:
                yPos = margin; // Reset Y to top margin for content
            }

            // --- File Content ---
            const fileNameText = `File ${index + 1}: ${file.file_name || "Unknown File"}`;
            const pathText = `Path: ${file.relative_path || "N/A"}`;
            const reqHeaderText = "Requirements:";
            const requirementsText = file.requirements || "No requirements provided.";

            // Add File Name
            doc.setFontSize(12);
            doc.setFont(undefined, "bold");
            doc.setTextColor(0, 128, 128); // Teal color for file name
            const fileNameHeight = doc.getTextDimensions(fileNameText, { maxWidth: maxLineWidth }).h;
            doc.text(fileNameText, margin, yPos, { maxWidth: maxLineWidth });
            yPos += fileNameHeight + 2; // Move down after file name

            // Add Path
            doc.setFontSize(10);
            doc.setFont(undefined, "normal");
            doc.setTextColor(100); // Gray color for path
            const pathHeight = doc.getTextDimensions(pathText, { maxWidth: maxLineWidth }).h;
            doc.text(pathText, margin, yPos, { maxWidth: maxLineWidth });
            yPos += pathHeight + 4; // Move down after path

            // Add Requirements Header
            doc.setFont(undefined, "bolditalic");
            doc.setTextColor(50); // Dark gray
            const reqHeaderHeight = doc.getTextDimensions(reqHeaderText, { maxWidth: maxLineWidth }).h;
            doc.text(reqHeaderText, margin, yPos, { maxWidth: maxLineWidth });
            yPos += reqHeaderHeight;

            // Add Requirements Text
            doc.setFont(undefined, "normal");
            doc.setTextColor(0); // Black for requirements text
            const requirementsLines = doc.splitTextToSize(requirementsText, maxLineWidth);
            // Check if requirements text itself needs pagination (unlikely for single file per page, but good practice)
            requirementsLines.forEach(line => {
                const lineHeight = doc.getTextDimensions(line, { maxWidth: maxLineWidth }).h * 1.15; // Add some spacing
                if (yPos + lineHeight > pageHeight - margin - 10) { // Check space before footer
                    doc.addPage();
                    yPos = margin; // Reset Y to top margin
                    // Optionally redraw header if desired on overflow pages
                }
                doc.text(line, margin, yPos);
                yPos += lineHeight;
            });
            // Add spacing after the requirements block if needed, before the next file (which starts on a new page)
            // yPos += 10;
        });

        // --- Footer on all pages ---
        const pageCount = doc.internal.getNumberOfPages();
        const generationDate = new Date().toLocaleDateString();
        for (let i = 1; i <= pageCount; i++) {
            doc.setPage(i); // Go to page i
            const footerY = pageHeight - margin + 8; // Position footer slightly below bottom margin
            doc.setFontSize(8);
            doc.setFont(undefined, "normal");
            doc.setTextColor(150);
            // Page number
            doc.text(`Page ${i} of ${pageCount}`, pageWidth / 2, footerY, { align: 'center' });
            // Generation date
            doc.text(`Generated on: ${generationDate}`, margin, footerY);
        }

        doc.save("CodebaseAnalysisReport_Paged.pdf"); // Changed filename
    };


    return (
        <div className="h-screen flex flex-col">
            <Navbar />
            <main className="flex flex-1 bg-gray-50">
                <div className="flex flex-col w-full h-full bg-white p-6 rounded shadow">
                    <h2 className="text-teal-700 text-xl font-semibold mb-4">
                        Analyze Codebase
                    </h2>
                    <form onSubmit={handleSubmit} className="space-y-4">
                        <div className="relative">
                            <label htmlFor="link" className="block text-gray-700 mb-1">
                                GitHub URL or Local Path
                            </label>
                            <div className="relative">
                                <input
                                    type="text"
                                    id="link"
                                    name="link"
                                    autoComplete="on"
                                    value={link}
                                    onChange={(e) => {
                                        setLink(e.target.value);
                                        // Clear status when input changes
                                        if (error) setError(null);
                                        // Don't clear results automatically on input change if they came from local storage initially
                                        // Only clear if user explicitly resets or starts a new analysis
                                        // if (analysisComplete) setAnalysisComplete(false);
                                        // if (result) setResult(null);
                                        // localStorage.removeItem("requirementsOutput");
                                    }}
                                    placeholder="e.g. https://github.com/you/repo or C:\path\to\project"
                                    className={`w-full border rounded px-3 py-2 focus:outline-none focus:ring-2 ${
                                        error
                                            ? "border-red-500 focus:ring-red-500"
                                            : "border-gray-300 focus:ring-teal-500"
                                    }`}
                                    required
                                />
                                {link && (
                                    <button
                                        type="button"
                                        onClick={() => {
                                            setLink("");
                                            // Optionally reset analysis if input is cleared manually
                                            // handleReset();
                                        }}
                                        className="absolute inset-y-0 right-0 flex items-center pr-2 text-gray-500 hover:text-gray-700"
                                        title="Clear input"
                                    >
                                        <XCircle className="h-5 w-5" />
                                    </button>
                                )}
                            </div>
                            {error && !loading && !analysisComplete && (
                                <p className="text-red-600 mt-1 text-xs flex items-center">
                                    <AlertCircle className="w-4 h-4 mr-1" /> {error}
                                </p>
                            )}
                        </div>

                        {/* Submit and Buttons Area */}
                        <div className="flex items-center space-x-3 pt-2 flex-wrap gap-y-2"> {/* Added flex-wrap and gap-y */}
                            <button
                                type="submit"
                                disabled={loading || !link}
                                className={`px-5 py-2 rounded text-white transition flex items-center justify-center min-w-[150px] ${
                                    loading || !link
                                        ? "bg-slate-400 cursor-not-allowed"
                                        : "bg-teal-700 hover:bg-teal-800"
                                }`}
                            >
                                {loading ? (
                                    <>
                                        <svg
                                            className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                                            xmlns="http://www.w3.org/2000/svg"
                                            fill="none"
                                            viewBox="0 0 24 24"
                                        >
                                            <circle
                                                className="opacity-25"
                                                cx="12"
                                                cy="12"
                                                r="10"
                                                stroke="currentColor"
                                                strokeWidth="4"
                                            ></circle>
                                            <path
                                                className="opacity-75"
                                                fill="currentColor"
                                                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                            ></path>
                                        </svg>
                                        Analyzing...
                                    </>
                                ) : (
                                    "Analyze Codebase"
                                )}
                            </button>

                            {/* Conditionally show View Results and Download PDF buttons */}
                            {result && analysisComplete && !loading && (
                                <>
                                    <button
                                        type="button"
                                        onClick={() => setShowResultsViewer(true)}
                                        className="px-5 py-2 rounded text-white bg-indigo-600 hover:bg-indigo-700 transition flex items-center"
                                    >
                                        <CheckCircle className="w-5 h-5 mr-2" /> View Results
                                    </button>
                                    <button
                                        type="button"
                                        onClick={downloadPDF}
                                        className="px-5 py-2 rounded text-white bg-green-600 hover:bg-green-700 transition flex items-center"
                                        title="Download PDF Report"
                                    >
                                        <Download className="w-5 h-5 mr-2" /> Download PDF
                                    </button>
                                </>
                            )}

                            {/* Reset Button */}
                            {/* Show reset button if there are results or analysis was completed */}
                            {(result || analysisComplete) && !loading && (
                                <button
                                    type="button"
                                    onClick={handleReset}
                                    className="px-5 py-2 rounded text-white bg-red-600 hover:bg-red-700 transition flex items-center"
                                    title="Clear current analysis results"
                                >
                                    <RotateCcw className="w-5 h-5 mr-2" /> Reset Analysis
                                </button>
                            )}
                        </div>
                        {/* Display general error message below buttons if analysis failed */}
                        {error && !loading && !analysisComplete && (
                             <p className="text-red-600 mt-2 text-sm flex items-center">
                                <AlertCircle className="w-5 h-5 mr-1 flex-shrink-0" /> {error}
                            </p>
                        )}
                    </form>

                    {/* Result Viewer Modal */}
                    {showResultsViewer && result && (
                        <Suspense
                            fallback={
                                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                                    <div className="text-white">Loading Results Viewer...</div>
                                </div>
                            }
                        >
                            <ResultViewer
                                isOpen={showResultsViewer}
                                onClose={() => setShowResultsViewer(false)}
                                // Pass result data to the viewer if it needs it directly
                                // analysisResult={result}
                            />
                        </Suspense>
                    )}
                </div>
            </main>
        </div>
    );
}