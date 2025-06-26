import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import axios from "../api/axiosConfig";
import Navbar from "../components/Navbar";
import { 
  FolderOpen, XCircle, AlertCircle, RotateCcw, CheckCircle, Download, 
  Zap, ArrowRight, History, Eye, Clock, FileText,
  Star, Activity, Loader2
} from 'lucide-react';
import { jsPDF } from "jspdf";
import autoTable from 'jspdf-autotable';

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
        setLoadingProgress(prev => {
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

  const isValidFolderName = (name) => /^[\w\-/.]+\/?$/.test(name) && name.trim().length > 0;

  const generateQuickStats = (data) => {
    if (!data || !data.requirements) return;
    
    const stats = {
      totalFiles: data.requirements.length,
      languages: [...new Set(data.requirements.map(f => f.file_name.split('.').pop()))],
      totalLines: data.requirements.reduce((acc, file) => acc + (file.requirements?.length || 0), 0),
      lastAnalyzed: new Date().toLocaleString()
    };
    setQuickStats(stats);
  };

  const saveToHistory = (folderName, result) => {
    const now = new Date();
    const historyItem = {
      id: Date.now(),
      folderName,
      timestamp: now.toISOString(),
      generatedTime: now.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      }),
      filesCount: result.requirements?.length || 0,
      status: 'completed',
      success: true,
      duration: Math.floor(Math.random() * 30) + 10 // Simulated duration in seconds
    };
    
    const updatedHistory = [historyItem, ...analysisHistory.slice(0, 9)]; // Keep last 10
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
    const updatedFavorites = favorites.filter(fav => fav !== folderName);
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
      setError("Enter valid S3 folder name (letters, numbers, slashes, hyphens, underscores, dots).");
      return;
    }

    setLoading(true);
    
    try {
      const res = await axios.post("/analysis", { folder_name: folderName });
      
      // Complete the progress bar
      setLoadingProgress(100);
      
      setTimeout(() => {
        // Patch: Always store the original S3 folder name and local path in the result object
        let localPath = res.data.local_path;
        if (!localPath) {
          // If backend doesn't return it, construct it (update this logic as needed)
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
      setError("Failed to analyze codebase. Please check the folder path and try again.");
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

  // Export functions
// Enhanced downloadPDF function with professional styling
// Enhanced downloadPDF function with professional styling
const downloadPDF = () => {
  if (!result || !result.requirements || !result.requirements.length) return;
  
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4',
  });

  // Set document properties
  doc.setProperties({
    title: `Codebase Analysis Report - ${folderName}`,
    author: 'App Discovery & Knowledge Management',
    creator: 'App Discovery System'
  });

  // Define margins and styles
  const marginLeft = 15;
  const marginTop = 20;
  const pageWidth = 210; // A4 width in mm
  const pageHeight = 297; // A4 height in mm
  const maxWidth = pageWidth - 2 * marginLeft;
  const footerY = pageHeight - 15; // Footer position from bottom

  // Color definitions
  const tealColor = [0, 128, 128]; // RGB for teal-700
  const blackColor = [0, 0, 0]; // RGB for black text
  const grayColor = [75, 85, 99]; // RGB for gray-600

  // Header
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(20);
  doc.setTextColor(...tealColor);
  doc.text('Business Logic Analysis Report', marginLeft, marginTop);
  
  // Draw header underline
  doc.setLineWidth(0.5);
  doc.setDrawColor(...tealColor);
  doc.line(marginLeft, marginTop + 2, pageWidth - marginLeft, marginTop + 2);
  
  // Draw document border
  doc.setLineWidth(0.3);
  doc.setDrawColor(...grayColor);
  doc.rect(10, 10, pageWidth - 20, pageHeight - 20, 'S'); // Border around content
  
  // Metadata
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(12);
  doc.setTextColor(...grayColor);
  const metadata = [
    `Generated on: ${new Date().toLocaleString()}`,
    `Folder: ${folderName}`,
    `Total Files: ${result.requirements.length}`,
    quickStats ? `Languages: ${quickStats.languages?.join(', ') || 'N/A'}` : ''
  ];
  
  let yPosition = marginTop + 15;
  metadata.forEach(line => {
    if (line) {
      doc.text(line, marginLeft, yPosition);
      yPosition += 8;
    }
  });

  // Add summary table with border
  yPosition += 10;
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(14);
  doc.setTextColor(...tealColor);
  doc.text('Summary', marginLeft, yPosition);
  yPosition += 8;

  const summaryData = [
    ['Total Files', result.requirements.length],
    ['Languages Detected', quickStats ? quickStats.languages?.join(', ') || 'N/A' : 'N/A'],
    ['Last Analyzed', quickStats ? quickStats.lastAnalyzed : 'N/A']
  ];

  autoTable(doc, {
    startY: yPosition,
    head: [['Metric', 'Value']],
    body: summaryData,
    theme: 'grid',
    styles: {
      font: 'helvetica',
      fontSize: 10,
      cellPadding: 3,
      textColor: [75, 85, 99], // Gray-600 for content
      lineColor: [200, 200, 200],
      lineWidth: 0.2
    },
    headStyles: {
      fillColor: tealColor,
      textColor: [255, 255, 255],
      fontStyle: 'bold'
    },
    margin: { left: marginLeft, right: marginLeft },
    tableLineColor: [75, 85, 99], // Gray-600 for table border
    tableLineWidth: 0.3
  });

  yPosition = doc.lastAutoTable.finalY + 15;

  // File Analysis Section
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(14);
  doc.setTextColor(...tealColor);
  doc.text('File Analysis', marginLeft, yPosition);
  yPosition += 8;

  result.requirements.forEach((file, index) => {
    // Start new page for each file (but check if content fits)
    if (index > 0 || yPosition > 250) { // Changed from marginTop to 250 to leave space for footer
      doc.addPage();
      yPosition = marginTop;
      // Draw border on new page
      doc.setLineWidth(0.3);
      doc.setDrawColor(...grayColor);
      doc.rect(10, 10, pageWidth - 20, pageHeight - 20, 'S');
    }

    // File header
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(12);
    doc.setTextColor(...tealColor);
    doc.text(`${index + 1}. ${file.file_name}`, marginLeft, yPosition);
    yPosition += 8;

    // File path
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10);
    doc.setTextColor(...grayColor);
    doc.text(`Path: ${file.relative_path || 'N/A'}`, marginLeft + 5, yPosition);
    yPosition += 8;

    // Requirements section with proper formatting
    if (file.requirements) {
      // Parse the requirements text to identify sections
      const requirementsText = file.requirements;
      const lines = requirementsText.split('\n');
      
      lines.forEach(line => {
        // Check if we need a new page - leave more space for footer
        if (yPosition > 250) { // Changed from 270 to 250
          doc.addPage();
          yPosition = marginTop;
          // Draw border on new page
          doc.setLineWidth(0.3);
          doc.setDrawColor(...grayColor);
          doc.rect(10, 10, pageWidth - 20, pageHeight - 20, 'S');
        }

        const trimmedLine = line.trim();
        
        if (trimmedLine) {
          // Check if line contains keywords that should be bold and teal
          const keywords = ['Overview', 'Objective', 'Use Case', 'Purpose', 'Description', 'Functionality', 'Features', 'Requirements', 'Key Functionalities', 'Workflow Summary', 'Dependent Files'];
          const isKeywordLine = keywords.some(keyword => 
            trimmedLine.startsWith(keyword) || 
            trimmedLine.includes(`${keyword}:`) ||
            trimmedLine.includes(`${keyword} `)
          );

          if (isKeywordLine) {
            // Split the line into keyword and content
            let keywordPart = '';
            let contentPart = '';
            
            for (const keyword of keywords) {
              if (trimmedLine.startsWith(keyword)) {
                const colonIndex = trimmedLine.indexOf(':');
                if (colonIndex !== -1) {
                  keywordPart = trimmedLine.substring(0, colonIndex + 1);
                  contentPart = trimmedLine.substring(colonIndex + 1).trim();
                } else {
                  keywordPart = keyword;
                  contentPart = trimmedLine.substring(keyword.length).trim();
                }
                break;
              }
            }

            // Print keyword in bold teal
            doc.setFont('helvetica', 'bold');
            doc.setFontSize(11);
            doc.setTextColor(...tealColor);
            const keywordWidth = doc.getTextWidth(keywordPart);
            doc.text(keywordPart, marginLeft + 5, yPosition);

            // Print content in normal gray
            if (contentPart) {
              doc.setFont('helvetica', 'normal');
              doc.setFontSize(10);
              doc.setTextColor(...grayColor);
              
              // Split content into multiple lines if needed
              const contentLines = doc.splitTextToSize(contentPart, maxWidth - 10 - keywordWidth - 5);
              doc.text(contentLines[0], marginLeft + 5 + keywordWidth + 3, yPosition);
              
              // Handle additional lines
              for (let i = 1; i < contentLines.length; i++) {
                yPosition += 5;
                if (yPosition > 250) { // Changed from 270 to 250
                  doc.addPage();
                  yPosition = marginTop;
                  // Draw border on new page
                  doc.setLineWidth(0.3);
                  doc.setDrawColor(...grayColor);
                  doc.rect(10, 10, pageWidth - 20, pageHeight - 20, 'S');
                }
                doc.text(contentLines[i], marginLeft + 5, yPosition);
              }
            }
          } else {
            // Regular content line
            doc.setFont('helvetica', 'normal');
            doc.setFontSize(10);
            doc.setTextColor(...grayColor);
            
            const textLines = doc.splitTextToSize(trimmedLine, maxWidth - 10);
            textLines.forEach((textLine, lineIndex) => {
              if (yPosition > 250) { // Changed from 270 to 250
                doc.addPage();
                yPosition = marginTop;
                // Draw border on new page
                doc.setLineWidth(0.3);
                doc.setDrawColor(...grayColor);
                doc.rect(10, 10, pageWidth - 20, pageHeight - 20, 'S');
              }
              doc.text(textLine, marginLeft + 5, yPosition);
              if (lineIndex < textLines.length - 1) yPosition += 5;
            });
          }
          
          yPosition += 6;
        } else {
          yPosition += 3; // Small space for empty lines
        }
      });
    } else {
      doc.setFont('helvetica', 'italic');
      doc.setFontSize(10);
      doc.setTextColor(...grayColor);
      doc.text('No requirements found', marginLeft + 5, yPosition);
      yPosition += 6;
    }

    yPosition += 10; // Space between sections
  });

  // Footer with professional styling - FIXED VERSION
  const pageCount = doc.internal.getNumberOfPages();
  const currentDate = new Date().toLocaleDateString();
  
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    
    // Set footer styling
    doc.setFontSize(9);
    doc.setTextColor(100, 100, 100);
    doc.setFont('helvetica', 'normal');
    
    // Company footer (left)
    doc.text(
      'App Discovery & Knowledge Management System',
      marginLeft,
      footerY
    );
    
    // Date (center) - calculate center position
    const dateText = `Generated: ${currentDate}`;
    const dateWidth = doc.getTextWidth(dateText);
    doc.text(
      dateText,
      (pageWidth - dateWidth) / 2,
      footerY
    );
    
    // Page number (right) - calculate right alignment
    const pageText = `Page ${i} of ${pageCount}`;
    const pageWidth_text = doc.getTextWidth(pageText);
    doc.text(
      pageText,
      pageWidth - marginLeft - pageWidth_text,
      footerY
    );
  }

  // Save the PDF with timestamp
  const timestamp = new Date().toISOString().split('T')[0];
  doc.save(`CodebaseAnalysis_${folderName.replace(/[/\\:*?"<>|]/g, '_')}_${timestamp}.pdf`);
};
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <Navbar />
      <div className="max-w-6xl mx-auto p-6">
        {/* Header Section */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-3">
            App Discovery & Knowledge Management
          </h1>
          <p className="text-lg text-gray-600 mb-6">
            Extract comprehensive business logic from your codebase with advanced analysis
          </p>
          
          {/* Quick Stats Bar */}
          {quickStats && !loading && (
            <div className="flex justify-center gap-8 text-sm text-gray-600 bg-white/60 backdrop-blur-sm rounded-xl p-4 max-w-3xl mx-auto shadow-sm">
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-teal-600" />
                <span className="font-medium">{quickStats.totalFiles} files analyzed</span>
              </div>
              <div className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-teal-600" />
                <span className="font-medium">{quickStats.languages.length} languages detected</span>
              </div>
              <div className="flex items-center gap-2">
                <Clock className="w-5 h-5 text-teal-600" />
                <span className="font-medium">{quickStats.lastAnalyzed}</span>
              </div>
            </div>
          )}
        </div>

        {/* Loading Overlay */}
        {loading && (
          <div className="fixed inset-0 bg-black/20 backdrop-blur-sm z-50 flex items-center justify-center">
            <div className="bg-white rounded-2xl p-8 shadow-2xl max-w-md w-full mx-4 border border-white/30">
              <div className="text-center">
                <div className="relative mb-6">
                  <Loader2 className="w-16 h-16 text-teal-600 animate-spin mx-auto" />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-8 h-8 bg-teal-100 rounded-full animate-pulse"></div>
                  </div>
                </div>
                
                <h3 className="text-xl font-semibold text-gray-800 mb-2">
                  Analyzing Your Codebase
                </h3>
                <p className="text-gray-600 mb-6">
                  Processing files and extracting business logic...
                </p>
                
                {/* Progress Bar */}
                <div className="w-full bg-gray-200 rounded-full h-3 mb-4 overflow-hidden">
                  <div 
                    className="bg-gradient-to-r from-teal-500 to-teal-600 h-3 rounded-full transition-all duration-500 ease-out"
                    style={{ width: `${loadingProgress}%` }}
                  >
                    <div className="h-full bg-gradient-to-r from-transparent via-white/30 to-transparent animate-pulse"></div>
                  </div>
                </div>
                
                <div className="text-sm text-gray-500">
                  {loadingProgress < 30 && "Connecting to repository..."}
                  {loadingProgress >= 30 && loadingProgress < 60 && "Scanning files..."}
                  {loadingProgress >= 60 && loadingProgress < 90 && "Analyzing code structure..."}
                  {loadingProgress >= 90 && "Finalizing results..."}
                </div>
                
                {/* Animated dots */}
                <div className="flex justify-center mt-4 space-x-1">
                  {[0, 1, 2].map((i) => (
                    <div
                      key={i}
                      className="w-2 h-2 bg-teal-500 rounded-full animate-bounce"
                      style={{ animationDelay: `${i * 0.2}s` }}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="grid lg:grid-cols-4 gap-6">
          {/* Main Analysis Panel */}
          <div className="lg:col-span-3">
            <div className="bg-white/90 backdrop-blur-sm p-8 rounded-2xl shadow-xl border border-white/30">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-semibold text-gray-800">Analysis Configuration</h2>
                <button
                  onClick={() => setShowHistory(!showHistory)}
                  className="text-sm text-teal-600 hover:text-teal-700 flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-teal-50 transition-colors"
                  disabled={loading}
                >
                  <History className="w-4 h-4" />
                  View History
                </button>
              </div>

              {/* Input Section */}
              <div className="space-y-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  S3 Folder Path
                </label>
                
                <div className="relative">
                  <FolderOpen className="absolute left-4 top-4 text-gray-400 z-10 w-5 h-5" />
                  <input
                    type="text"
                    value={folderName}
                    onChange={(e) => { 
                      setFolderName(e.target.value); 
                      if (error) setError(null); 
                    }}
                    disabled={loading}
                    className="w-full pl-12 pr-24 py-4 border-2 border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent transition-all text-lg disabled:bg-gray-100 disabled:text-gray-500"
                    placeholder="your-folder-name/ or path/to/project"
                  />
                  <div className="absolute right-4 top-4 flex items-center gap-2">
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
                    <div className="text-sm text-gray-500 mb-3 font-medium">Favorite Projects:</div>
                    <div className="flex flex-wrap gap-2">
                      {favorites.map((fav, index) => (
                        <button
                          key={index}
                          onClick={() => setFolderName(fav)}
                          className="px-4 py-2 bg-teal-50 text-teal-700 rounded-lg text-sm hover:bg-teal-100 transition-colors font-medium border border-teal-200"
                        >
                          {fav}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Error Display */}
                {error && (
                  <div className="mt-4 p-4 bg-red-50 border-l-4 border-red-400 rounded-lg">
                    <p className="text-sm text-red-700 flex items-center">
                      <AlertCircle className="w-5 h-5 mr-3 flex-shrink-0" />
                      {error}
                    </p>
                  </div>
                )}

                {/* Submit Button */}
                <button
                  onClick={handleSubmit}
                  disabled={loading || !folderName}
                  className="w-full bg-gradient-to-r from-teal-600 to-teal-700 hover:from-teal-700 hover:to-teal-800 disabled:from-gray-400 disabled:to-gray-500 text-white py-4 rounded-xl flex justify-center items-center gap-3 font-semibold text-lg transition-all transform hover:scale-[1.02] disabled:scale-100 shadow-lg"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-6 h-6 animate-spin" />
                      Analyzing Codebase...
                    </>
                  ) : (
                    <>
                      <Zap className="w-5 h-5" /> 
                      Start Advanced Analysis 
                      <ArrowRight className="w-5 h-5" />
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Results Section */}
            {analysisComplete && result && !loading && (
              <div className="mt-6 bg-white/90 backdrop-blur-sm p-8 rounded-2xl shadow-xl border border-white/30">
                <div className="flex justify-between items-center mb-6">
                  <div className="flex items-center gap-4">
                    <CheckCircle className="text-green-600 w-8 h-8" />
                    <div>
                      <h3 className="text-xl font-semibold text-gray-800">Analysis Complete</h3>
                      <p className="text-gray-600">
                        {result.requirements?.length馆length || 0} files processed successfully
                      </p>
                    </div>
                  </div>
                  <button 
                    onClick={handleReset} 
                    className="text-sm text-teal-700 hover:text-teal-800 flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-teal-50 font-medium transition-colors"
                  >
                    <RotateCcw className="w-4 h-4" /> 
                    New Analysis
                  </button>
                </div>

                {/* Action Buttons */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <button
                    onClick={() => navigate('/results', { state: { result } })}
                    className="bg-teal-600 hover:bg-teal-700 text-white py-4 rounded-xl flex justify-center items-center gap-3 font-semibold transition-all shadow-lg hover:shadow-xl"
                  >
                    <Eye className="w-5 h-5" />
                    View Results
                  </button>

                  <button
                    onClick={downloadPDF}
                    className="bg-teal-600 hover:bg-teal-700 text-white py-4 rounded-xl flex justify-center items-center gap-3 font-semibold transition-all shadow-lg hover:shadow-xl"
                  >
                    <Download className="w-5 h-5" /> 
                    Download PDF
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-6">
            {/* Analysis History */}
            {showHistory && !loading && (
              <div className="bg-white/90 backdrop-blur-sm p-6 rounded-2xl shadow-xl border border-white/30">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
                    <History className="w-5 h-5 text-teal-600" />
                    Recent Analysis
                  </h3>
                  <button
                    onClick={() => setShowHistory(false)}
                    className="text-gray-400 hover:text-gray-600 transition-colors"
                  >
                    <XCircle className="w-5 h-5" />
                  </button>
                </div>
                <div className="space-y-3 max-h-80 overflow-y-auto">
                  {analysisHistory.length > 0 ? (
                    analysisHistory.map((item) => (
                      <div
                        key={item.id}
                        onClick={() => loadFromHistory(item)}
                        className="p-4 bg-gray-50 hover:bg-gray-100 rounded-xl cursor-pointer transition-colors border border-gray-200 group"
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div className="font-medium text-sm text-gray-800 truncate flex-1 pr-2">
                            {item.folderName}
                          </div>
                          <div className="flex items-center gap-1">
                            <CheckCircle className="w-3 h-3 text-green-500" />
                            <span className="text-xs text-green-600 font-medium">Success</span>
                          </div>
                        </div>
                        <div className="text-xs text-gray-600 space-y-1">
                          <div className="flex justify-between items-center">
                            <span className="flex items-center gap-1">
                              <FileText className="w-3 h-3" />
                              {item.filesCount} files
                            </span>
                            <span className="flex items-center gap-1">
                              <Clock className="w-3 h-3" />
                              {item.duration || 'N/A'}s
                            </span>
                          </div>
                          <div className="text-xs text-gray-500 mt-1">
                            Generated: {item.generatedTime || new Date(item.timestamp).toLocaleString()}
                          </div>
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-gray-500 text-center py-8">No previous analyses</p>
                  )}
                </div>
              </div>
            )}

            {/* Quick Stats */}
            {quickStats && !loading && (
              <div className="bg-white/90 backdrop-blur-sm p-6 rounded-2xl shadow-xl border border-white/30">
                <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-teal-600" />
                  Analysis Overview
                </h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Total Files:</span>
                    <span className="text-lg font-bold text-teal-600">{quickStats.totalFiles}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Languages:</span>
                    <span className="text-lg font-bold text-teal-600">{quickStats.languages.length}</span>
                  </div>
                  <div className="pt-3 border-t border-gray-200">
                    <div className="text-xs text-gray-500 mb-2 font-medium">Detected Languages:</div>
                    <div className="flex flex-wrap gap-2">
                      {quickStats.languages.map((lang, index) => (
                        <span key={index} className="px-3 py-1 bg-teal-100 text-teal-700 rounded-full text-xs font-medium">
                          {lang}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Pro Tips */}
            {!loading && (
              <div className="bg-gradient-to-br from-teal-50 to-blue-50 p-6 rounded-2xl border border-teal-100 shadow-lg">
                <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-teal-600" />
                  Pro Tips
                </h3>
                <ul className="text-sm text-gray-700 space-y-2">
                  <li className="flex items-start gap-2">
                    <span className="text-teal-600 font-bold">•</span>
                    Use specific folder paths for more accurate results
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-teal-600 font-bold">•</span>
                    Star frequently analyzed projects for quick access
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-teal-600 font-bold">•</span>
                    Download PDF reports for detailed documentation
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-teal-600 font-bold">•</span>
                    Check history to revisit previous analyses
                  </li>
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}