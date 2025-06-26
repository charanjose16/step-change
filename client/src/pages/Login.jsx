import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axiosInstance from "../api/axiosConfig";
import Carousel from "../components/Carousel";
import logo from "../assets/icons/logo.png";
import { Eye, EyeOff } from 'lucide-react';

export default function Login({ onLoginSuccess }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [activeTab, setActiveTab] = useState('signin');
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
        <div className="w-full max-w-sm bg-white rounded-lg shadow-sm border border-gray-200">
          <div className="p-6">
            {/* Header */}
            <div className="text-center mb-6">
              <img src={logo} alt="Logo" className="w-12 h-12 mx-auto mb-3" />
              <h1 className="text-2xl font-bold text-gray-900 mb-1">
                Welcome Back
              </h1>
              <p className="text-gray-600 text-sm">
                Sign in to continue your conversation journey
              </p>
            </div>

            {/* Tab Navigation */}
            <div className="flex mb-4 bg-gray-100 rounded-lg p-1">
              <button
                onClick={() => setActiveTab('signin')}
                className={`flex-1 py-2 px-3 rounded-md text-sm font-medium transition-all ${
                  activeTab === 'signin'
                    ? 'bg-white text-teal-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                Sign In
              </button>
              <button
                onClick={() => setActiveTab('signup')}
                className={`flex-1 py-2 px-3 rounded-md text-sm font-medium transition-all ${
                  activeTab === 'signup'
                    ? 'bg-white text-teal-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                Sign Up
              </button>
            </div>

            {/* Form Content */}
            {activeTab === 'signin' ? (
              <div>
                {errorMessage && (
                  <p className="text-red-500 text-sm mb-4">
                    {errorMessage}
                  </p>
                )}

                {/* Form */}
                <div className="space-y-4">
                  {/* Username Field */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Username
                    </label>
                    <input
                      type="text"
                      value={username}
                      onChange={(e) => setUsername(e.target.value)}
                      placeholder="Enter your username"
                      required
                      className="w-full px-3 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 placeholder-gray-400"
                    />
                  </div>

                  {/* Password Field */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Password
                    </label>
                    <div className="relative">
                      <input
                        type={showPassword ? "text" : "password"}
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="Enter your password"
                        required
                        className="w-full px-3 py-2.5 pr-10 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 placeholder-gray-400"
                      />
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
                      >
                        {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                      </button>
                    </div>
                  </div>

                  {/* Remember Me and Forgot Password */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        id="remember"
                        checked={rememberMe}
                        onChange={(e) => setRememberMe(e.target.checked)}
                        className="h-4 w-4 text-teal-600 focus:ring-teal-500 border-gray-300 rounded"
                      />
                      <label htmlFor="remember" className="ml-2 text-sm text-gray-600">
                        Remember me
                      </label>
                    </div>
                    <button type="button" className="text-sm text-teal-600 hover:text-teal-700">
                      Forgot password?
                    </button>
                  </div>

                  {/* Sign In Button */}
                  <button
                    onClick={handleSubmit}
                    disabled={loading}
                    className="w-full bg-teal-600 text-white py-2.5 px-4 rounded-lg hover:bg-teal-700 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:ring-offset-2 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                  >
                    {loading ? (
                      <svg
                        className="animate-spin h-5 w-5"
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
                      <>
                        Sign In
                        <span className="ml-2">→</span>
                      </>
                    )}
                  </button>
                </div>

                {/* Sign Up Link */}
                <div className="text-center mt-4">
                  <span className="text-gray-600 text-sm">Don't have an account? </span>
                  <button 
                    type="button" 
                    onClick={() => setActiveTab('signup')}
                    className="text-teal-600 hover:text-teal-700 font-medium text-sm"
                  >
                    Sign up here
                  </button>
                </div>
              </div>
            ) : (
              <div>
                {/* Sign Up Form */}
                <div className="space-y-4">
                  {/* Name Field */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Full Name
                    </label>
                    <input
                      type="text"
                      placeholder="Enter your full name"
                      className="w-full px-3 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 placeholder-gray-400"
                    />
                  </div>

                  {/* Email Field */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Email Address
                    </label>
                    <input
                      type="email"
                      placeholder="Enter your email"
                      className="w-full px-3 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 placeholder-gray-400"
                    />
                  </div>

                  {/* Password Field */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Password
                    </label>
                    <div className="relative">
                      <input
                        type={showPassword ? "text" : "password"}
                        placeholder="Create a password"
                        className="w-full px-3 py-2.5 pr-10 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 placeholder-gray-400"
                      />
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-700 hover:text-gray-600"
                      >
                        {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                      </button>
                    </div>
                  </div>

                  {/* Sign Up Button */}
                  <button
                    type="button"
                    className="w-full bg-teal-600 text-white py-2.5 px-4 rounded-lg hover:bg-teal-700 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:ring-offset-2 transition-colors font-medium flex items-center justify-center"
                  >
                    Sign Up
                    <span className="ml-2">→</span>
                  </button>
                </div>

                {/* Sign In Link */}
                <div className="text-center mt-4">
                  <span className="text-gray-700 text-sm">Already have an account? </span>
                  <button 
                    type="button" 
                    onClick={() => setActiveTab('signin')}
                    className="text-teal-700 hover:text-teal-700 font-medium text-sm"
                  >
                    Sign in here
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}