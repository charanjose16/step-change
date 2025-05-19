import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axiosInstance from "../api/axiosConfig";
import Carousel from "../components/Carousel";
import logo from "../assets/icons/logo.png"

export default function Login({ onLoginSuccess }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setErrorMessage("");
    try {
      const { data } = await axiosInstance.post("/auth/login", {
        username,
        password,
      });
      // Save the token in local storage
      localStorage.setItem("access_token", data.access_token);
      // Optionally update axiosInstance default headers for future requests
      axiosInstance.defaults.headers.common.Authorization = `Bearer ${data.access_token}`;
      onLoginSuccess();
      navigate("/dashboard");
    } catch (error) {
      setErrorMessage("Login failed. Please check your credentials.");
      console.error(error);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen grid grid-cols-1 md:grid-cols-2">
      {/* Left Side – Carousel (visible on medium and larger screens) */}
      <div className="hidden md:block">
        <Carousel />
      </div>

      {/* Right Side – Login Form */}
      <div className="flex items-center justify-center bg-gray-50 p-6">
        <div className="w-full max-w-sm bg-white p-8 rounded shadow-lg">
        <div className="text-center mb-6">
            <img src={logo} alt="Logo" className="w-16 h-16 mx-auto mb-4" />
            <h1 className="text-2xl  font-bold text-teal-700 mb-2">
                Code Assessment
            </h1>
            <p className="text-gray-600 mt-2">
                Generate Code assesments for scala 
            </p>
        </div>
          {errorMessage && (
            <p className="text-red-500 text-sm text-left mb-2">
              {errorMessage}
            </p>
          )}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Username
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                className="mt-1 block w-full border border-gray-300 rounded p-2 focus:outline-none focus:ring-teal-500 focus:border-teal-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="mt-1 block w-full border border-gray-300 rounded p-2 focus:outline-none focus:ring-teal-500 focus:border-teal-500"
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-teal-600 text-white py-2 rounded hover:bg-teal-700 transition duration-300"
            >
              {loading ? (
                <svg
                  className="animate-spin h-5 w-5 mx-auto"
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
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                  />
                </svg>
              ) : (
                "Sign In"
              )}
            </button>
          </form>
          
        </div>
      </div>
    </div>
  );
}
