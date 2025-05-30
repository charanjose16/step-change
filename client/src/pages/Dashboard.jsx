
import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "../api/axiosConfig";
import Navbar from "../components/Navbar";
import { XCircle, CheckCircle, AlertCircle, Download, RotateCcw, FolderOpen, Code, FileText } from "lucide-react";
import { jsPDF } from "jspdf";

export default function Dashboard() {
  const [folderName, setFolderName] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const storedResult = localStorage.getItem("requirementsOutput");
    if (storedResult) {
      try {
        const parsedResult = JSON.parse(storedResult);
        setResult(parsedResult);
        setAnalysisComplete(true);
      } catch (e) {
        console.error("Failed to parse stored results:", e);
        localStorage.removeItem("requirementsOutput");
      }
    }
  }, []);

  const isValidFolderName = (name) => {
    return /^[\w\-/]+\/?$/.test(name) && name.trim().length > 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setAnalysisComplete(false);
    localStorage.removeItem("requirementsOutput");

    if (!isValidFolderName(folderName)) {
      setError("Please enter a valid S3 folder name (alphanumeric, slashes, hyphens, underscores).");
      return;
    }

    setLoading(true);

    try {
      const res = await axios.post("/analysis", { folder_name: folderName });
      console.log("Response:", res.data);
      setResult(res.data);
      localStorage.setItem("requirementsOutput", JSON.stringify(res.data));
      setAnalysisComplete(true);
      setError(null);
    } catch (err) {
      console.error(err);
      const errMsg =
        err.response && err.response.data && err.response.data.detail
          ? err.response.data.detail
          : "Failed to analyze codebase.";
      setError(errMsg);
      setResult(null);
      setAnalysisComplete(false);
      localStorage.removeItem("requirementsOutput");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setAnalysisComplete(false);
    setError(null);
    setFolderName("");
    localStorage.removeItem("requirementsOutput");
    console.log("Analysis results cleared.");
  };

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
    let yPos;

    const drawHeader = (docInstance, currentY) => {
      docInstance.setFontSize(18);
      docInstance.setFont(undefined, "bold");
      docInstance.setTextColor(40, 58, 90);
      docInstance.text("Codebase Analysis Report", margin, currentY);
      currentY += 6;
      docInstance.setDrawColor(40, 58, 90);
      docInstance.line(margin, currentY, pageWidth - margin, currentY);
      currentY += 12;
      return currentY;
    };

    yPos = margin + 8;
    yPos = drawHeader(doc, yPos);

    result.requirements.forEach((file, index) => {
      if (index > 0) {
        doc.addPage();
        yPos = margin + 8;
        yPos = margin;
      }

      const fileNameText = `File ${index + 1}: ${file.file_name || "Unknown File"}`;
      const pathText = `Path: ${file.relative_path || "N/A"}`;
      const reqHeaderText = "Requirements:";
      const requirementsText = file.requirements || "No requirements provided.";

      doc.setFontSize(12);
      doc.setFont(undefined, "bold");
      doc.setTextColor(0, 128, 128);
      const fileNameHeight = doc.getTextDimensions(fileNameText, { maxWidth: maxLineWidth }).h;
      doc.text(fileNameText, margin, yPos, { maxWidth: maxLineWidth });
      yPos += fileNameHeight + 2;

      doc.setFontSize(10);
      doc.setFont(undefined, "normal");
      doc.setTextColor(100);
      const pathHeight = doc.getTextDimensions(pathText, { maxWidth: maxLineWidth }).h;
      doc.text(pathText, margin, yPos, { maxWidth: maxLineWidth });
      yPos += pathHeight + 4;

      doc.setFont(undefined, "bolditalic");
      doc.setTextColor(50);
      const reqHeaderHeight = doc.getTextDimensions(reqHeaderText, { maxWidth: maxLineWidth }).h;
      doc.text(reqHeaderText, margin, yPos, { maxWidth: maxLineWidth });
      yPos += reqHeaderHeight;

      doc.setFont(undefined, "normal");
      doc.setTextColor(0);
      const requirementsLines = doc.splitTextToSize(requirementsText, maxLineWidth);

      requirementsLines.forEach((line) => {
        const lineHeight = doc.getTextDimensions(line, { maxWidth: maxLineWidth }).h * 1.15;
        if (yPos + lineHeight > pageHeight - margin - 10) {
          doc.addPage();
          yPos = margin;
        }
        doc.text(line, margin, yPos);
        yPos += lineHeight;
      });
    });

    const pageCount = doc.internal.getNumberOfPages();
    const generationDate = new Date().toLocaleDateString();
    for (let i = 1; i <= pageCount; i++) {
      doc.setPage(i);
      const footerY = pageHeight - margin + 8;
      doc.setFontSize(8);
      doc.setFont(undefined, "normal");
      doc.setTextColor(150);
      doc.text(`Page ${i} of ${pageCount}`, pageWidth / 2, footerY, { align: "center" });
      doc.text(`Generated on: ${generationDate}`, margin, footerY);
    }

    doc.save("CodebaseAnalysisReport.pdf");
  };

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <Navbar />
      <div className="flex-1 container mx-auto px-4 py-6">
        {/* Header/Input Form Section */}
        <div className="bg-gradient-to-br from-teal-700 to-teal-600 rounded-2xl shadow-lg overflow-hidden">
          <div className="p-6">
            <div className="flex items-center mb-4">
              <Code className="h-10 w-10 text-teal-200 mr-3" />
              <h1 className="text-3xl font-bold text-white">Code to Business Logic Converter</h1>
            </div>
            <p className="text-teal-100 text-sm mb-4 max-w-2xl leading-relaxed">
              Transform your codebase into clear business requirements. Enter an S3 bucket folder name to start.
            </p>

            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="bg-white/10 backdrop-blur-sm p-6 rounded-2xl border border-teal-300/20 shadow-lg">
                <div className="flex flex-col md:flex-row gap-4">
                  <div className="flex-1">
                    <label htmlFor="folderName" className="block text-teal-100 font-semibold text-sm mb-2">
                      S3 Folder Name
                    </label>
                    <div className="relative">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <FolderOpen className="h-5 w-5 text-teal-300" />
                      </div>
                      <input
                        type="text"
                        id="folderName"
                        name="folderName"
                        autoComplete="off"
                        value={folderName}
                        onChange={(e) => {
                          setFolderName(e.target.value);
                          if (error) setError(null);
                        }}
                        placeholder="Enter S3 folder name (e.g., stepchange-testing-repo-main/)"
                        className={`w-full bg-white/20 backdrop-blur-sm border text-white pl-10 pr-10 py-3 rounded-lg focus:outline-none focus:ring-2 text-sm ${
                          error
                            ? "border-red-400 focus:ring-red-400"
                            : "border-teal-300/40 focus:ring-teal-300"
                        }`}
                        required
                        aria-label="S3 bucket folder name"
                      />
                      {folderName && (
                        <button
                          type="button"
                          onClick={() => setFolderName("")}
                          className="absolute inset-y-0 right-0 flex items-center pr-3 text-teal-300 hover:text-white transition-colors"
                          title="Clear input"
                          aria-label="Clear input"
                        >
                          <XCircle className="h-5 w-5" />
                        </button>
                      )}
                    </div>
                    {error && !loading && !analysisComplete && (
                      <p className="text-red-400 mt-2 text-xs flex items-center">
                        <AlertCircle className="w-4 h-4 mr-1 flex-shrink-0" /> {error}
                      </p>
                    )}
                  </div>

                  <div className="flex items-end">
                    <button
                      type="submit"
                      disabled={loading || !folderName}
                      className={`h-12 px-6 rounded-lg text-white font-semibold text-sm transition-all duration-200 flex items-center justify-center shadow-lg hover:scale-105 ${
                        loading || !folderName
                          ? "bg-teal-400/50 cursor-not-allowed"
                          : "bg-teal-600 hover:bg-teal-500 hover:shadow-teal-500/40"
                      }`}
                      aria-label="Generate business logic"
                    >
                      {loading ? (
                        <>
                          <svg
                            className="animate-spin -ml-2 mr-2 h-5 w-5 text-white"
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
                        "Generate Business Logic"
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </form>
          </div>
        </div>

        {analysisComplete && result && (
          <div className="mt-6 animate-fadeIn">
            {/* Analysis Summary and Detailed Results */}
            <div className="bg-white rounded-2xl shadow-lg p-4 hover:shadow-2xl transition-shadow duration-300">
              <div className="flex items-center justify-between border-b border-teal-200 pb-3 mb-4">
                <div className="flex items-center">
                  <CheckCircle className="h-6 w-6 text-green-500 mr-2" />
                  <h2 className="text-xl font-bold text-teal-800">Analysis Results</h2>
                  <div className="flex items-center bg-teal-50 rounded-xl p-1 shadow-sm hover:shadow-xl transition-shadow duration-300 ml-5">
                    <div className="h-8 w-8 rounded-full bg-teal-100 flex items-center justify-center mr-3">
                      <FileText className="h-4 w-4 text-teal-600" />
                    </div>
                    <div>
                      <div className="font-semibold text-teal-800 text-xs">Files Analyzed</div>
                      <div className="text-sm font-semibold text-teal-600">{result.requirements?.length || 0} <span className="text-sm mr-2">Code files processed</span></div>
                    </div>
                  </div>
                </div>
                <button
                  onClick={handleReset}
                  className="flex items-center px-3 py-1 bg-teal-100 hover:bg-teal-200 text-teal-800 rounded-lg transition-all duration-200 text-xs font-semibold shadow-lg hover:scale-105"
                  aria-label="Reset analysis results"
                >
                  <RotateCcw className="w-4 h-4 mr-1" />
                  Reset
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <p className="text-teal-600 text-sm md:col-span-3 mx-auto max-w-prose leading-relaxed text-center">
                  Explore the business logic extracted from each file for a comprehensive understanding of your codebase.
                </p>

                <div className="flex justify-center md:col-span-3">
                  <button
                    onClick={() => navigate('/results', { state: { result } })}
                    className="flex items-center justify-center px-4 py-2 bg-teal-600 hover:bg-teal-500 text-white rounded-lg transition-all duration-200 shadow-lg hover:shadow-teal-500/40 hover:scale-105 text-sm font-semibold"
                    aria-label="View detailed analysis results"
                  >
                    <Code className="w-4 h-4 mr-1" />
                    View Detailed Results
                  </button>
                </div>
              </div>
            </div>

            {/* Analysis PDF Download Section */}
            <div className="bg-white rounded-2xl shadow-lg p-6 mt-4 hover:shadow-2xl transition-shadow duration-300">
              <div className="flex items-center border-b border-teal-200 pb-3 mb-4">
                <Download className="h-5 w-5 text-emerald-600 mr-2" />
                <h2 className="text-xl font-bold text-teal-800">Export Documentation</h2>
              </div>
              <p className="text-teal-600 text-sm mb-4 mx-auto max-w-prose leading-relaxed text-center">
                Download a PDF report with all business logic requirements for documentation and sharing.
              </p>
              <div className="flex justify-center">
                <button
                  onClick={downloadPDF}
                  className="flex items-center px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg transition-all duration-200 shadow-lg hover:shadow-emerald-500/40 hover:scale-105 text-sm font-semibold"
                  aria-label="Download PDF report"
                >
                  <Download className="w-4 h-4 mr-1" />
                  Download PDF
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
