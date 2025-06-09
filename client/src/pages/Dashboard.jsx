 
import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "../api/axiosConfig";
import Navbar from "../components/Navbar";
import {
  Code,
  FolderOpen,
  XCircle,
  AlertCircle,
  CheckCircle,
  FileText,
  RotateCcw,
  Download,
  Sparkles,
  ArrowRight,
  Zap
} from 'lucide-react';
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
    <div className="min-h-screen bg-gray-50">
      <Navbar />
     
      <div className="container mx-auto px-4 py-8">
        {/* Hero Section */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center bg-teal-50 text-teal-700 px-4 py-2 rounded-full text-sm font-medium mb-4">
       
           
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
           
            <span className="text-teal-700">  App Discovery and Knowledge Management</span>
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Automatically extract clear business requirements from your codebase using advanced AI analysis
          </p>
        </div>
 
        {/* Main Input Card */}
        <div className="max-w-4xl mx-auto">
          <div className="bg-white rounded-3xl shadow-xl border border-gray-100 overflow-hidden">
            {/* Card Header */}
            <div className="bg-gradient-to-r from-teal-700 to-teal-600 px-8 py-6">
              <div className="flex items-center text-white">
                <div className="h-12 w-12 bg-white/20 rounded-xl flex items-center justify-center mr-4">
                  <FolderOpen className="h-6 w-6" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold">Code To Business Logic Converter </h2>
                  <p className="text-teal-100 mt-1">Enter your S3 bucket folder to begin analysis</p>
                </div>
              </div>
            </div>
 
            {/* Form Section */}
            <div className="p-8">
              <div className="space-y-6">
                <div>
                  <label htmlFor="folderName" className="block text-sm font-semibold text-gray-700 mb-3">
                    S3 Folder Path
                  </label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                      <FolderOpen className="h-5 w-5 text-gray-400" />
                    </div>
                    <input
                      type="text"
                      id="folderName"
                      value={folderName}
                      onChange={(e) => {
                        setFolderName(e.target.value);
                        if (error) setError(null);
                      }}
                      placeholder="stepchange-testing-repo-main/"
                      className={`w-full bg-gray-50 border-2 text-gray-900 pl-12 pr-12 py-4 rounded-xl focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent text-lg transition-all ${
                        error ? "border-red-300 bg-red-50" : "border-gray-200 hover:border-gray-300"
                      }`}
                    />
                    {folderName && (
                      <button
                        type="button"
                        onClick={() => setFolderName("")}
                        className="absolute inset-y-0 right-0 flex items-center pr-4 text-gray-400 hover:text-gray-600 transition-colors"
                      >
                        <XCircle className="h-5 w-5" />
                      </button>
                    )}
                  </div>
                  {error && (
                    <div className="mt-3 flex items-center text-red-600 text-sm">
                      <AlertCircle className="w-4 h-4 mr-2 flex-shrink-0" />
                      {error}
                    </div>
                  )}
                </div>
 
                <button
                  onClick={handleSubmit}
                  disabled={loading || !folderName}
                  className={`w-full py-4 px-6 rounded-xl text-white font-semibold text-lg transition-all duration-200 flex items-center justify-center ${
                    loading || !folderName
                      ? "bg-gray-300 cursor-not-allowed"
                      : "bg-teal-700 hover:bg-teal-800 shadow-lg hover:shadow-xl transform hover:scale-[1.02]"
                  }`}
                >
                  {loading ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-3 h-6 w-6 text-white" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Analyzing Repository...
                    </>
                  ) : (
                    <>
                      <Zap className="w-5 h-5 mr-2" />
                      Start Analysis
                      <ArrowRight className="w-5 h-5 ml-2" />
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
 
        {/* Results Section */}
        {analysisComplete && result && (
          <div className="max-w-4xl mx-auto mt-8 space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Success Card */}
            <div className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden">
              <div className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center">
                    <div className="h-12 w-12 bg-green-100 rounded-xl flex items-center justify-center mr-4">
                      <CheckCircle className="h-6 w-6 text-green-600" />
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-gray-900">Analysis Complete</h3>
                      <p className="text-gray-600">Successfully processed your repository</p>
                    </div>
                  </div>
                  <button
                    onClick={handleReset}
                    className="flex items-center px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-all duration-200 font-medium"
                  >
                    <RotateCcw className="w-4 h-4 mr-2" />
                    New Analysis
                  </button>
                </div>
 
                {/* Stats */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="bg-teal-50 rounded-xl p-4 text-center">
                    <div className="h-10 w-10 bg-teal-700 rounded-lg flex items-center justify-center mx-auto mb-2">
                      <FileText className="h-5 w-5 text-white" />
                    </div>
                    <div className="text-2xl font-bold text-teal-700">{result.requirements?.length || 0}</div>
                    <div className="text-sm text-teal-600 font-medium">Files Analyzed</div>
                  </div>
                  <div className="bg-gray-50 rounded-xl p-4 text-center">
                    <div className="h-10 w-10 bg-gray-700 rounded-lg flex items-center justify-center mx-auto mb-2">
                      <Code className="h-5 w-5 text-white" />
                    </div>
                    <div className="text-2xl font-bold text-gray-700">100%</div>
                    <div className="text-sm text-gray-600 font-medium">Success Rate</div>
                  </div>
                  <div className="bg-blue-50 rounded-xl p-4 text-center">
                    <div className="h-10 w-10 bg-blue-700 rounded-lg flex items-center justify-center mx-auto mb-2">
                      <Sparkles className="h-5 w-5 text-white" />
                    </div>
                    <div className="text-2xl font-bold text-blue-700">AI</div>
                    <div className="text-sm text-blue-600 font-medium">Powered</div>
                  </div>
                </div>
 
                <div className="flex justify-center">
                  <button
                    onClick={() => navigate('/results', { state: { result } })}
                    className="flex items-center px-6 py-3 bg-teal-700 hover:bg-teal-800 text-white rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105 font-semibold"
                  >
                    <Code className="w-5 h-5 mr-2" />
                    View Detailed Results
                    <ArrowRight className="w-5 h-5 ml-2" />
                  </button>
                </div>
              </div>
            </div>
 
            {/* Export Card */}
            <div className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden">
              <div className="p-6">
                <div className="flex items-center mb-4">
                  <div className="h-10 w-10 bg-teal-100 rounded-lg flex items-center justify-center mr-3">
                    <Download className="h-5 w-5 text-teal-700" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-gray-900">Export Documentation</h3>
                    <p className="text-gray-600">Download comprehensive business logic report</p>
                  </div>
                </div>
               
                <div className="bg-gray-50 rounded-xl p-4 mb-4">
                  <p className="text-sm text-gray-700 text-center">
                    Get a professionally formatted PDF containing all extracted business requirements,
                    ready for stakeholder review and documentation.
                  </p>
                </div>
 
                <div className="flex justify-center">
                  <button
                    onClick={downloadPDF}
                    className="flex items-center px-6 py-3 bg-gradient-to-r from-teal-700 to-teal-600 hover:from-teal-800 hover:to-teal-700 text-white rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105 font-semibold"
                  >
                    <Download className="w-5 h-5 mr-2" />
                    Download PDF Report
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
 