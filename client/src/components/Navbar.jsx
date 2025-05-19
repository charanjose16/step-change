import React from "react";
import { Link } from "react-router-dom";
import UstLogo from "../assets/ust-white-logo.svg";
import { LogOut, User } from "lucide-react";
import { jwtDecode } from "jwt-decode";

export default function Navbar() {
    const token = localStorage.getItem("access_token");
    let username = "User";

    if (token) {
        try {
            const decodedToken = jwtDecode(token);
            username = decodedToken.username || decodedToken.sub || "User";
        } catch (error) {
            console.error("Error decoding token:", error);
        }
    }

    const handleLogout = () => {
        localStorage.removeItem("access_token");
        window.location.reload(); // Consider using react-router's navigation for a smoother SPA experience
    };

    return (
        // Increased shadow for more depth
        <header className="bg-teal-700 text-white shadow-lg w-full">
            {/* Increased vertical padding */}
            <nav className="flex items-center justify-between py-5 px-6">
                <div className="flex items-center space-x-4">
                    {/* Logo hover effect remains */}
                    <div className="mr-4 transform transition duration-300 hover:scale-105">
                        <img src={UstLogo} alt="UST Logo" className="w-10 h-10" />
                    </div>
                    {/* Removed hover text color change */}
                    <Link
                        to="/dashboard"
                        className="px-3 py-2 rounded transition duration-300" // Removed hover:text-teal-200
                    >
                        Dashboard
                    </Link>
                </div>
                {/* Increased spacing between user info and logout button */}
                <div className="flex items-center space-x-6">
                    {/* User info section with hover effect */}
                    <div className="flex items-center space-x-2 cursor-default px-3 py-2 rounded transition duration-300 hover:bg-teal-600">
                        <User className="h-5 w-5" />
                        <span className="hidden sm:inline">{username}</span>
                    </div>
                    {/* Updated Logout button style: removed border, added bg hover */}
                    <button
                        onClick={handleLogout}
                        className="flex items-center space-x-2 px-4 py-2 rounded transition duration-300 text-white hover:bg-teal-600 hover:shadow-sm" // Removed border classes, added bg hover
                        title="Logout" // Added title for accessibility
                    >
                        <LogOut className="h-5 w-5" />
                        {/* Kept span for consistency, can remove sm:inline if always visible */}
                        <span className="hidden sm:inline">Logout</span>
                    </button>
                </div>
            </nav>
        </header>
    );
}