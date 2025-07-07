import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import axios from "../api/axiosConfig";
import Navbar from "../components/Navbar";

import {
  FolderOpen,
  XCircle,
  AlertCircle,
  RotateCcw,
  CheckCircle,
  Download,
  Zap,
  ArrowRight,
  History,
  Eye,
  Clock,
  FileText,
  Star,
  Activity,
  Loader2,
  CloudDownload,
  Brain,
} from "lucide-react";
import { jsPDF } from "jspdf";
import autoTable from "jspdf-autotable";

export default function Dashboard() {
  // State management
  const [folderName, setFolderName] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [favorites, setFavorites] = useState([]);
  const [quickStats, setQuickStats] = useState(null);
  const [loadingProgress, setLoadingProgress] = useState(0);

  const navigate = useNavigate();
  const fileInputRef = useRef(null);

  // Load data from localStorage on component mount
  useEffect(() => {
    loadStoredData();
  }, []);

  // Simulate loading progress for better UX
  useEffect(() => {
    let interval;
    if (loading) {
      setLoadingProgress(0);
      interval = setInterval(() => {
        setLoadingProgress((prev) => {
          if (prev >= 90) return prev;
          return prev + Math.random() * 15;
        });
      }, 500);
    } else {
      setLoadingProgress(0);
    }
    return () => clearInterval(interval);
  }, [loading]);

  // Helper functions
  const loadStoredData = () => {
    const storedResult = localStorage.getItem("requirementsOutput");
    const storedHistory = localStorage.getItem("analysisHistory");
    const storedFavorites = localStorage.getItem("favorites");

    if (storedResult) {
      try {
        const parsedResult = JSON.parse(storedResult);
        setResult(parsedResult);
        setAnalysisComplete(true);
        generateQuickStats(parsedResult);
      } catch {
        localStorage.removeItem("requirementsOutput");
      }
    }

    if (storedHistory) {
      try {
        setAnalysisHistory(JSON.parse(storedHistory));
      } catch {
        localStorage.removeItem("analysisHistory");
      }
    }

    if (storedFavorites) {
      try {
        setFavorites(JSON.parse(storedFavorites));
      } catch {
        localStorage.removeItem("favorites");
      }
    }
  };

  const isValidFolderName = (name) =>
    /^[\w\-/.]+\/?$/.test(name) && name.trim().length > 0;

  const generateQuickStats = (data) => {
    if (!data || !data.requirements) return;

    const stats = {
      totalFiles: data.requirements.length,
      languages: [...new Set(data.requirements.map((f) => f.file_name.split(".").pop()))],
      totalLines: data.requirements.reduce(
        (acc, file) => acc + (file.requirements?.length || 0),
        0
      ),
      lastAnalyzed: new Date().toLocaleString(),
    };
    setQuickStats(stats);
  };

  const saveToHistory = (folderName, result) => {
    const now = new Date();
    const historyItem = {
      id: Date.now(),
      folderName,
      timestamp: now.toISOString(),
      generatedTime: now.toLocaleString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      }),
      filesCount: result.requirements?.length || 0,
      status: "completed",
      success: true,
      duration: Math.floor(Math.random() * 30) + 10,
    };

    const updatedHistory = [historyItem, ...analysisHistory.slice(0, 9)];
    setAnalysisHistory(updatedHistory);
    localStorage.setItem("analysisHistory", JSON.stringify(updatedHistory));
  };

  const addToFavorites = (folderName) => {
    if (!favorites.includes(folderName)) {
      const updatedFavorites = [...favorites, folderName];
      setFavorites(updatedFavorites);
      localStorage.setItem("favorites", JSON.stringify(updatedFavorites));
    }
  };

  const removeFromFavorites = (folderName) => {
    const updatedFavorites = favorites.filter((fav) => fav !== folderName);
    setFavorites(updatedFavorites);
    localStorage.setItem("favorites", JSON.stringify(updatedFavorites));
  };

  // Main handlers
  const handleSubmit = async (e) => {
    e.preventDefault();

    setError(null);
    setResult(null);
    setAnalysisComplete(false);
    setQuickStats(null);
    setLoadingProgress(0);
    localStorage.removeItem("requirementsOutput");

    if (!isValidFolderName(folderName)) {
      setError(
        "Enter a valid S3 folder path (letters, numbers, slashes, hyphens, underscores, dots)."
      );
      return;
    }

    setLoading(true);

    try {
      const res = await axios.post("/analysis", { folder_name: folderName });

      setLoadingProgress(100);

      setTimeout(() => {
        let localPath = res.data.local_path;
        if (!localPath) {
          localPath = `/tmp/${folderName}`;
        }
        const resultWithFolderName = { ...res.data, folder_name: folderName, local_path: localPath };
        setResult(resultWithFolderName);
        localStorage.setItem("requirementsOutput", JSON.stringify(resultWithFolderName));
        setAnalysisComplete(true);
        generateQuickStats(res.data);
        saveToHistory(folderName, res.data);
        setLoading(false);
      }, 500);
    } catch (err) {
      setError(
        "Failed to fetch from S3 bucket or generate business logic. Please verify the folder path and try again."
      );
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFolderName("");
    setResult(null);
    setError(null);
    setAnalysisComplete(false);
    setQuickStats(null);
    setLoadingProgress(0);
    localStorage.removeItem("requirementsOutput");
  };

  const loadFromHistory = (historyItem) => {
    setFolderName(historyItem.folderName);
    setShowHistory(false);
  };

  // Download PDF function
  const downloadPDF = () => {
    if (!result || !result.requirements || !result.requirements.length) return;

    const doc = new jsPDF({
      orientation: "portrait",
      unit: "mm",
      format: "a4",
    });

    doc.setProperties({
      title: `Codebase Analysis Report - ${folderName}`,
      author: "App Discovery & Knowledge Management",
      creator: "App Discovery System",
    });

    const marginLeft = 15;
    const marginTop = 20;
    const pageWidth = 210;
    const pageHeight = 297;
    const maxWidth = pageWidth - 2 * marginLeft;
    const footerY = pageHeight - 15;

    const tealColor = [0, 128, 128];
    const grayColor = [75, 85, 99];

    doc.setFont("helvetica", "bold");
    doc.setFontSize(20);
    doc.setTextColor(...tealColor);
    doc.text("Business Logic Analysis Report", marginLeft, marginTop);

    doc.setLineWidth(0.5);
    doc.setDrawColor(...tealColor);
    doc.line(marginLeft, marginTop + 2, pageWidth - marginLeft, marginTop + 2);

    doc.setLineWidth(0.3);
    doc.setDrawColor(...grayColor);
    doc.rect(10, 10, pageWidth - 20, pageHeight - 20, "S");

    doc.setFont("helvetica", "normal");
    doc.setFontSize(12);
    doc.setTextColor(...grayColor);
    const metadata = [
      `Generated on: ${new Date().toLocaleString()}`,
      `Folder: ${folderName}`,
      `Total Files: ${result.requirements.length}`,
      quickStats ? `Languages: ${quickStats.languages?.join(", ") || "N/A"}` : "",
    ];

    let yPosition = marginTop + 15;
    metadata.forEach((line) => {
      if (line) {
        doc.text(line, marginLeft, yPosition);
        yPosition += 8;
      }
    });

    yPosition += 10;
    doc.setFont("helvetica", "bold");
    doc.setFontSize(14);
    doc.setTextColor(...tealColor);
    doc.text("Summary", marginLeft, yPosition);
    yPosition += 8;

    const summaryData = [
      ["Total Files", result.requirements.length],
      ["Languages Detected", quickStats ? quickStats.languages?.join(", ") || "N/A" : "N/A"],
      ["Last Analyzed", quickStats ? quickStats.lastAnalyzed : "N/A"],
    ];

    autoTable(doc, {
      startY: yPosition,
      head: [["Metric", "Value"]],
      body: summaryData,
      theme: "grid",
      styles: {
        font: "helvetica",
        fontSize: 10,
        cellPadding: 3,
        textColor: [75, 85, 99],
        lineColor: [200, 200, 200],
        lineWidth: 0.2,
      },
      headStyles: {
        fillColor: tealColor,
        textColor: [255, 255, 255],
        fontStyle: "bold",
      },
      margin: { left: marginLeft, right: marginLeft },
      tableLineColor: [75, 85, 99],
      tableLineWidth: 0.3,
    });

    yPosition = doc.lastAutoTable.finalY + 15;

    doc.setFont("helvetica", "bold");
    doc.setFontSize(14);
    doc.setTextColor(...tealColor);
    doc.text("File Analysis", marginLeft, yPosition);
    yPosition += 8;

    result.requirements.forEach((file, index) => {
      if (index > 0 || yPosition > 250) {
        doc.addPage();
        yPosition = marginTop;
        doc.setLineWidth(0.3);
        doc.setDrawColor(...grayColor);
        doc.rect(10, 10, pageWidth - 20, pageHeight - 20, "S");
      }

      doc.setFont("helvetica", "bold");
      doc.setFontSize(12);
      doc.setTextColor(...tealColor);
      doc.text(`${index + 1}. ${file.file_name}`, marginLeft, yPosition);
      yPosition += 8;

      doc.setFont("helvetica", "normal");
      doc.setFontSize(10);
      doc.setTextColor(...grayColor);
      doc.text(`Path: ${file.relative_path || "N/A"}`, marginLeft + 5, yPosition);
      yPosition += 8;

      if (file.requirements) {
        const requirementsText = file.requirements;
        const lines = requirementsText.split("\n");

        lines.forEach((line) => {
          if (yPosition > 250) {
            doc.addPage();
            yPosition = marginTop;
            doc.setLineWidth(0.3);
            doc.setDrawColor(...grayColor);
            doc.rect(10, 10, pageWidth - 20, pageHeight - 20, "S");
          }

          const trimmedLine = line.trim();

          if (trimmedLine) {
            const keywords = [
              "Integrates with frontend framework for setup.",
              "Overview",
              "Objective",
              "Use Case",
              "Purpose",
              "Description",
              "Functionality",
              "Features",
              "Requirements",
              "Key Functionalities",
              "Workflow Summary",
              "Dependent Files",
            ];
            const isKeywordLine = keywords.some(
              (keyword) =>
                trimmedLine.startsWith(keyword) ||
                trimmedLine.includes(`${keyword}:`) ||
                trimmedLine.includes(`${keyword} `)
            );

            if (isKeywordLine) {
              let keywordPart = "";
              let contentPart = "";

              for (const keyword of keywords) {
                if (trimmedLine.startsWith(keyword)) {
                  keywordPart = keyword;
                  contentPart = trimmedLine.substring(keyword.length).trim();
                  break;
                }
              }

              doc.setFont("helvetica", "bold");
              doc.setFontSize(11);
              doc.setTextColor(...tealColor);
              const keywordWidth = doc.getTextWidth(keywordPart);
              doc.text(keywordPart, marginLeft + 5, yPosition);

              if (contentPart) {
                doc.setFont("helvetica", "normal");
                doc.setFontSize(10);
                doc.setTextColor(...grayColor);

                if (keywordPart === "Dependent Files") {
                  const fileItems = contentPart.split("\n").map((item) => item.trim());
                  fileItems.forEach((item, index) => {
                    if (index > 0) yPosition += 6;
                    if (yPosition > 250) {
                      doc.addPage();
                      yPosition = marginTop;
                      doc.setLineWidth(0.3);
                      doc.setDrawColor(...grayColor);
                      doc.rect(10, 10, pageWidth - 20, pageHeight - 20, "S");
                    }
                    doc.text(item, marginLeft + 20, yPosition);
                  });
                } else {
                  const contentLines = doc.splitTextToSize(contentPart, maxWidth - 10 - keywordWidth - 5);
                  doc.text(contentLines[0], marginLeft + 5 + keywordWidth + 3, yPosition);
                  for (let i = 1; i < contentLines.length; i++) {
                    yPosition += 5;
                    if (yPosition > 250) {
                      doc.addPage();
                      yPosition = marginTop;
                      doc.setLineWidth(0.3);
                      doc.setDrawColor(...grayColor);
                      doc.rect(10, 10, pageWidth - 20, pageHeight - 20, "S");
                    }
                    doc.text(contentLines[i], marginLeft + 5, yPosition);
                  }
                }
              }
            } else {
              doc.setFont("helvetica", "normal");
              doc.setFontSize(10);
              doc.setTextColor(...grayColor);

              const textLines = doc.splitTextToSize(trimmedLine, maxWidth - 10);
              textLines.forEach((textLine, lineIndex) => {
                if (yPosition > 250) {
                  doc.addPage();
                  yPosition = marginTop;
                  doc.setLineWidth(0.3);
                  doc.setDrawColor(...grayColor);
                  doc.rect(10, 10, pageWidth - 20, pageHeight - 20, "S");
                }
                doc.text(textLine, marginLeft + 5, yPosition);
                if (lineIndex < textLines.length - 1) yPosition += 5;
              });
            }

            yPosition += 6;
          } else {
            yPosition += 3;
          }
        });
      } else {
        doc.setFont("helvetica", "italic");
        doc.setFontSize(10);
        doc.setTextColor(...grayColor);
        doc.text("No requirements found", marginLeft + 5, yPosition);
        yPosition += 6;
      }

      yPosition += 10;
    });

    const pageCount = doc.internal.getNumberOfPages();
    const currentDate = new Date().toLocaleDateString();

    for (let i = 1; i <= pageCount; i++) {
      doc.setPage(i);

      doc.setFontSize(9);
      doc.setTextColor(100, 100, 100);
      doc.setFont("helvetica", "normal");

      doc.text("App Discovery & Knowledge Management System", marginLeft, footerY);

      const dateText = `Generated: ${currentDate}`;
      const dateWidth = doc.getTextWidth(dateText);
      doc.text(dateText, (pageWidth - dateWidth) / 2, footerY);

      const pageText = `Page ${i} of ${pageCount}`;
      const pageWidth_text = doc.getTextWidth(pageText);
      doc.text(pageText, pageWidth - marginLeft - pageWidth_text, footerY);
    }

    const timestamp = new Date().toISOString().split("T")[0];
    doc.save(`CodebaseAnalysis_${folderName.replace(/[/\\:*?"<>|]/g, "_")}_${timestamp}.pdf`);
  };

  // Steps for loading overlay
  const steps = [
    {
      id: 1,
      label: "Fetching from S3",
      description: "Securely retrieving codebase assets from the configured Amazon S3 bucket.",
      iconActive: "/src/assets/icons/1.png.png",
      iconInactive: "/src/assets/icons/1.png.png",
    },
    {
      id: 2,
      label: "Analyzing Files",
      description: "Performing structural and dependency analysis on the retrieved codebase.",
      iconActive: "/src/assets/icons/2.png.png",
      iconInactive: "/src/assets/icons/2.png.png",
    },
    {
      id: 3,
      label: "Generating Business Logic",
      description: "Extracting actionable business rules and functional insights from application logic.",
      iconActive: "/src/assets/icons/3.png.png",
      iconInactive: "/src/assets/icons/3.png.png",
    },
    {
      id: 4,
      label: "Finalizing",
      description: "Compiling a comprehensive report based on the processed data for review.",
      iconActive: "/src/assets/icons/4.png.png",
      iconInactive: "/src/assets/icons/4.png.png",
    },
  ];
  
  
  const currentStep = Math.floor(loadingProgress / 45) + 1;
  

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-100 to-teal-50 font-sans">
      <Navbar />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 h-[calc(100vh-70px)] overflow-y-auto">
        

        {/* Header Section */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 tracking-tight">
            App Discovery & Knowledge Management
          </h1>
          <p className="mt-2 text-sm text-gray-600 font-medium">
            Extract and analyze business logic from your S3-hosted codebase
          </p>
        </div>

        {/* Main Content Layout */}
        <div className="flex gap-6">
          {/* Main Analysis Panel */}
          <div className="w-full">
            <div className="bg-gradient-to-br from-white via-cream-100 to-cream-50 p-6 rounded-xl shadow-md border border-gray-200">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold text-gray-800">Analysis Configuration</h2>
                <button
                  onClick={() => setShowHistory(!showHistory)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg font-semibold text-sm transition-all duration-300 ${
                    showHistory
                      ? "bg-teal-700 text-white hover:bg-teal-800"
                      : "text-teal-700 hover:text-teal-800 hover:bg-teal-50 border border-teal-200"
                  }`}
                  disabled={loading}
                >
                  <History className="w-5 h-5" />
                  {showHistory ? "Hide History" : "View History"}
                </button>
              </div>

              {/* Input Section */}
              <div className="space-y-4">
                <label className="block text-sm font-semibold text-gray-700">
                  S3 Folder Path
                </label>

                <div className="relative">
                  <FolderOpen className="absolute left-4 top-3.5 text-gray-400 w-5 h-5" />
                  <input
                    type="text"
                    value={folderName}
                    onChange={(e) => {
                      setFolderName(e.target.value);
                      if (error) setError(null);
                    }}
                    disabled={loading}
                    className="w-full pl-12 pr-20 py-3 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-700 focus:border-transparent text-sm font-medium disabled:bg-gray-100 disabled:text-gray-500 transition-all"
                    placeholder="s3://bucket-name/path/to/project"
                  />
                  <div className="absolute right-4 top-3.5 flex items-center gap-3">
                    {favorites.includes(folderName) ? (
                      <Star
                        className="w-5 h-5 text-yellow-500 cursor-pointer hover:text-yellow-600"
                        onClick={() => !loading && removeFromFavorites(folderName)}
                        fill="currentColor"
                      />
                    ) : (
                      folderName && (
                        <Star
                          className="w-5 h-5 text-gray-400 hover:text-yellow-500 cursor-pointer transition-colors"
                          onClick={() => !loading && addToFavorites(folderName)}
                        />
                      )
                    )}
                    {folderName && (
                      <XCircle
                        className="w-5 h-5 text-gray-400 hover:text-gray-600 cursor-pointer transition-colors"
                        onClick={() => !loading && setFolderName("")}
                      />
                    )}
                  </div>
                </div>

                {/* Favorites Quick Access */}
                {favorites.length > 0 && !loading && (
                  <div className="mt-4">
                    <div className="text-sm text-gray-600 font-semibold mb-3">Favorite Paths:</div>
                    <div className="flex flex-wrap gap-3">
                      {favorites.map((fav, index) => (
                        <button
                          key={index}
                          onClick={() => setFolderName(fav)}
                          className="px-4 py-2 bg-teal-50 text-teal-700 rounded-lg text-sm font-semibold hover:bg-teal-100 transition-all border border-teal-200 shadow-sm"
                        >
                          {fav}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Error Display */}
                {error && (
                  <div className="mt-4 p-4 bg-red-50 border-l-4 border-red-500 rounded-lg">
                    <p className="text-sm text-red-700 flex items-center font-medium">
                      <AlertCircle className="w-5 h-5 mr-2 flex-shrink-0" />
                      {error}
                    </p>
                  </div>
                )}

                {/* Submit Button */}
                <button
                  onClick={handleSubmit}
                  disabled={loading || !folderName}
                  className="w-full bg-teal-700 hover:bg-teal-800 disabled:bg-gray-400 text-white py-3 rounded-lg flex justify-center items-center gap-3 font-semibold text-sm transition-all duration-300 hover:shadow-lg disabled:shadow-none"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Zap className="w-5 h-5" />
                      Analyze S3 Codebase
                      <ArrowRight className="w-5 h-5" />
                    </>
                  )}
                </button>
              </div>

              {/* Generation Section */}
              {loading && (
  <div className="mt-6 p-6 bg-white rounded-2xl shadow-xl border border-green-200">
    <h3 className="text-xl font-bold text-green-800 mb-5 flex items-center gap-3">
      <Activity className="w-6 h-6 animate-pulse" />
      Analysis in Progress
    </h3>

    <div className="relative pl-10">
      {steps.map((step, index) => {
        const isCompleted = step.id < currentStep;
        const isActive = step.id === currentStep;

        return (
          <div key={step.id} className="relative mb-6 last:mb-0">
            {/* Connector line */}
            {index < steps.length - 1 && (
              <div
                className={`absolute left-5 top-8 w-[2px] z-0 ${
                  isCompleted ? "bg-green-500" : "bg-gray-300"
                }`}
                style={{ height: "calc(100% + 12px)" }}
              />
            )}

            {/* Step */}
            <div
              className={`relative z-10 flex items-start gap-4 py-2 transition-all duration-300 ${
                isCompleted
                  ? "text-green-900"
                  : isActive
                  ? "text-green-700"
                  : "text-gray-400"
              }`}
            >
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center border-2 shadow-sm transition-all duration-300 ${
                  isCompleted
                    ? "bg-green-100 border-green-600"
                    : isActive
                    ? "bg-green-50 border-green-500 animate-pulse"
                    : "bg-gray-50 border-gray-300"
                }`}
              >
                {isCompleted ? (
                  <CheckCircle className="w-6 h-6 text-green-600" />
                ) : (
                  <img
                    src={isActive ? step.iconActive : step.iconInactive}
                    alt={step.label}
                    className="w-8 h-8"
                  />
                )}
              </div>

              <div className="flex-1">
                <span className="text-base font-bold leading-snug text-black">
                  {step.label}
                </span>
                <p className="text-sm mt-1 text-gray-800 font-medium leading-relaxed tracking-tight">
                  {step.description}
                </p>
              </div>
            </div>
          </div>
                      );
                    })}
                    <div className="mt-4 text-right text-sm text-green-700 font-semibold pr-4">
                      
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Results Section */}
            {analysisComplete && result && !loading && (
              <div className="mt-6 bg-white/95 backdrop-blur-sm p-8 rounded-2xl shadow-lg border border-gray-100">
                <div className="flex justify-between items-center mb-6">
                  <div className="flex items-center gap-4">
                    <CheckCircle className="text-teal-700 w-8 h-8" />
                    <div>
                      <h3 className="text-xl font-bold text-gray-900">Business Logic Generated</h3>
                      <p className="text-sm text-gray-600 font-medium">
                        Successfully processed {result.requirements?.length || 0} files
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={handleReset}
                    className="text-sm text-teal-700 hover:text-teal-800 flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-teal-50 font-semibold transition-all border border-teal-200"
                  >
                    <RotateCcw className="w-5 h-5" />
                    Start New Analysis
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <button
                    onClick={() => navigate("/results", { state: { result } })}
                    className="bg-teal-700 hover:bg-teal-800 text-white py-3 rounded-lg flex justify-center items-center gap-3 font-semibold text-sm transition-all hover:shadow-lg"
                  >
                    <Eye className="w-5 h-5" />
                    View Business Logic
                  </button>

                  <button
                    onClick={downloadPDF}
                    className="bg-teal-700 hover:bg-teal-800 text-white py-3 rounded-lg flex justify-center items-center gap-3 font-semibold text-sm transition-all hover:shadow-lg"
                  >
                    <Download className="w-5 h-5" />
                    Download Report
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Sidebar with Slide Animation */}
          <div
            className={`transition-all duration-500 ease-in-out ${
              showHistory ? "w-1/3 opacity-100 translate-x-0" : "w-0 opacity-0 translate-x-full"
            } overflow-hidden`}
          >
            {showHistory && !loading && (
              <div className="bg-white/95 backdrop-blur-sm p-6 rounded-2xl shadow-lg border border-gray-100 h-full">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                    <History className="w-5 h-5 text-teal-700" />
                    Analysis History
                  </h3>
                  <button
                    onClick={() => setShowHistory(false)}
                    className="text-gray-400 hover:text-gray-600 transition-colors"
                  >
                    <XCircle className="w-5 h-5" />
                  </button>
                </div>
                <div className="space-y-3 max-h-[calc(100vh-200px)] overflow-y-auto pr-2">
                  {analysisHistory.length > 0 ? (
                    analysisHistory.map((item) => (
                      <div
                        key={item.id}
                        onClick={() => loadFromHistory(item)}
                        className="p-4 bg-gray-50 hover:bg-gray-100 rounded-lg cursor-pointer transition-all border border-gray-200 hover:shadow-sm group"
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div className="font-semibold text-sm text-gray-900 truncate flex-1 pr-3">
                            {item.folderName}
                          </div>
                          <div className="flex items-center gap-2">
                            <CheckCircle className="w-4 h-4 text-teal-700" />
                            <span className="text-xs text-teal-700 font-semibold">Completed</span>
                          </div>
                        </div>
                        <div className="text-sm text-gray-600 space-y-1">
                          <div className="flex justify-between items-center">
                            <span className="flex items-center gap-2">
                              <FileText className="w-4 h-4" />
                              {item.filesCount} Files
                            </span>
                            <span className="flex items-center gap-2">
                              <Clock className="w-4 h-4" />
                              {item.duration || "N/A"}s
                            </span>
                          </div>
                          <div className="text-xs text-gray-500 font-medium">
                            {item.generatedTime || new Date(item.timestamp).toLocaleString()}
                          </div>
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-gray-500 text-center py-8 font-medium">
                      No previous analyses found
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}