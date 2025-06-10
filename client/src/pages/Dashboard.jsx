// Simplified version of the Dashboard component with loader
import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "../api/axiosConfig";
import Navbar from "../components/Navbar";
import { FolderOpen, XCircle, AlertCircle, RotateCcw, CheckCircle, Download, Zap, ArrowRight } from 'lucide-react';
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
      } catch {
        localStorage.removeItem("requirementsOutput");
      }
    }
  }, []);

  const isValidFolderName = (name) => /^[\w\-/]+\/?$/.test(name) && name.trim().length > 0;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setAnalysisComplete(false);
    localStorage.removeItem("requirementsOutput");

    if (!isValidFolderName(folderName)) {
      setError("Enter valid S3 folder name (letters, slashes, hyphens, underscores).")
      return;
    }

    setLoading(true);
    try {
      const res = await axios.post("/analysis", { folder_name: folderName });
      setResult(res.data);
      localStorage.setItem("requirementsOutput", JSON.stringify(res.data));
      setAnalysisComplete(true);
    } catch (err) {
      setError("Failed to analyze codebase.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFolderName("");
    setResult(null);
    setError(null);
    setAnalysisComplete(false);
    localStorage.removeItem("requirementsOutput");
  };

  const downloadPDF = () => {
    if (!result || !result.requirements || !result.requirements.length) return;
    const doc = new jsPDF();
    doc.setFontSize(16).text("Business Logic Report", 20, 20);
    result.requirements.forEach((file, i) => {
      doc.addPage();
      doc.setFontSize(12).text(`File ${i + 1}: ${file.file_name}`, 20, 40);
      doc.setFontSize(10).text(`Path: ${file.relative_path}`, 20, 50);
      doc.text(file.requirements || "No requirements found.", 20, 60);
    });
    doc.save("CodebaseAnalysisReport.pdf");
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <div className="max-w-xl mx-auto p-4">
        <h1 className="text-2xl font-bold text-center text-teal-700 mb-4">App Discovery And Knowledge Managment</h1>
        <p className="text-center text-gray-600 mb-6">Get business logic from your codebase</p>

        <div className="bg-white p-6 rounded-xl shadow">
          <label className="block mb-2 text-sm font-medium">S3 Folder Path</label>
          <div className="relative">
            <FolderOpen className="absolute left-3 top-3 text-gray-400" />
            <input
              type="text"
              value={folderName}
              onChange={(e) => { setFolderName(e.target.value); if (error) setError(null); }}
              className="w-full pl-10 pr-10 py-2 border rounded-md focus:outline-none focus:ring-2"
              placeholder="your-folder-name/"
            />
            {folderName && (
              <XCircle className="absolute right-3 top-3 text-gray-400 cursor-pointer" onClick={() => setFolderName("")} />
            )}
          </div>
          {error && <p className="text-sm text-red-500 mt-2 flex items-center"><AlertCircle className="w-4 h-4 mr-1" />{error}</p>}

          <button
            onClick={handleSubmit}
            disabled={loading || !folderName}
            className="mt-4 w-full bg-teal-600 hover:bg-teal-700 text-white py-2 rounded-md flex justify-center items-center gap-2"
          >
            {loading ? (
              <>
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                </svg>
                Analyzing...
              </>
            ) : (
              <>
                <Zap className="w-4 h-4" /> Start Analysis <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </div>

        {analysisComplete && result && (
          <div className="mt-6 bg-white p-6 rounded-xl shadow space-y-4">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-3">
                <CheckCircle className="text-green-600" />
                <span className="font-semibold text-gray-800">Analysis Complete</span>
              </div>
              <button onClick={handleReset} className="text-sm text-teal-700 flex items-center gap-1">
                <RotateCcw className="w-4 h-4" /> Reset
              </button>
            </div>

            <p className="text-sm text-gray-600">Files Analyzed: <strong>{result.requirements?.length || 0}</strong></p>

            <button
              onClick={() => navigate('/results', { state: { result } })}
              className="w-full bg-teal-600 hover:bg-teal-700 text-white py-2 rounded-md flex justify-center items-center gap-2"
            >
              View Detailed Results <ArrowRight className="w-4 h-4" />
            </button>

            <button
              onClick={downloadPDF}
              className="w-full bg-gray-800 hover:bg-gray-900 text-white py-2 rounded-md flex justify-center items-center gap-2"
            >
              <Download className="w-4 h-4" /> Download PDF Report
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
