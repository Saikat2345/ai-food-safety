import React, { useState, useEffect } from 'react';
import { Play, Apple, TrendingUp, Zap, Shield, Users, ChevronRight, Menu, X, Star } from 'lucide-react';

const NutritionHero = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [scrollY, setScrollY] = useState(0);
  const handleStart = () => {
    window.location.href = "http://localhost:5173/analyze"; // Replace with your actual start URL
  }
  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const features = [
    {
      icon: TrendingUp,
      title: "Smart Analytics",
      description: "AI-powered insights that adapt to your lifestyle and goals"
    },
    {
      icon: Apple,
      title: "Food Recognition",
      description: "Snap a photo and instantly log your meals with precision"
    },
    {
      icon: Zap,
      title: "Real-time Tracking",
      description: "Live updates on your nutrition goals throughout the day"
    }
  ];

  const stats = [
    { number: "3+", label: "Active Users" },
    { number: "10+", label: "Substances Logged" },
    { number: "4.9★", label: "App Rating" },
    { number: "20+", label: "Calories Tracked" }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 text-white overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-emerald-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-1/2 -left-40 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute bottom-0 right-1/4 w-64 h-64 bg-purple-500/10 rounded-full blur-3xl animate-pulse delay-2000"></div>
      </div>

      {/* Navigation */}
      <nav className="relative z-50 px-6 py-6">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-10 h-10 bg-gradient-to-br from-emerald-500 to-cyan-500 rounded-xl flex items-center justify-center">
              <Apple className="w-6 h-6 text-white" />
            </div>
            <span className="text-2xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
              NutriX
            </span>
          </div>

          {/* Desktop Menu */}
          <div className="hidden md:flex items-center space-x-8">
            <button className="px-6 py-2 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full hover:from-emerald-600 hover:to-cyan-600 transition-all transform hover:scale-105 cursor-pointer"
            onClick={handleStart}>
              Get Started
            </button>
          </div>

          {/* Mobile Menu Button */}
          <button 
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            className="md:hidden p-2 rounded-lg bg-gray-800/50 backdrop-blur-lg"
          >
            {isMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>

        {/* Mobile Menu */}
        {isMenuOpen && (
          <div className="md:hidden absolute top-full left-0 right-0 bg-gray-900/95 backdrop-blur-lg border-t border-gray-700/50 p-6">
            <div className="space-y-4">
              <a href="#features" className="block text-gray-300 hover:text-white transition-colors">Features</a>
              <a href="#pricing" className="block text-gray-300 hover:text-white transition-colors">Pricing</a>
              <a href="#about" className="block text-gray-300 hover:text-white transition-colors">About</a>
              <button className="w-full px-6 py-3 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full hover:from-emerald-600 hover:to-cyan-600 transition-all">
                Get Started
              </button>
            </div>
          </div>
        )}
      </nav>

      {/* Hero Section */}
      <div className="relative z-10 max-w-7xl mx-auto px-6 pt-20 pb-32">
        <div className="text-center">
          {/* Badge */}
          <div className="inline-flex items-center space-x-2 bg-gray-800/50 backdrop-blur-lg border border-gray-700/50 rounded-full px-4 py-2 mb-8">
            <span className="text-sm text-gray-300">An app for those who care about themselves</span>
          </div>

          {/* Main Headline */}
          <h1 className="text-5xl md:text-7xl font-bold leading-tight mb-6">
            <span className="bg-gradient-to-r from-white via-gray-100 to-gray-300 bg-clip-text text-transparent">
              Transform Your
            </span>
            <br />
            <span className="bg-gradient-to-r from-emerald-400 via-cyan-400 to-blue-400 bg-clip-text text-transparent">
              Nutrition Journey
            </span>
          </h1>

          {/* Subtitle */}
          <p className="text-xl md:text-2xl text-gray-400 max-w-3xl mx-auto mb-12 leading-relaxed">
            Effortlessly track your nutrition with AI-powered insights, personalized recommendations, 
            and a community that supports your wellness goals.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6 mb-16">
            <button className="group px-8 py-4 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full hover:from-emerald-600 hover:to-cyan-600 transition-all transform hover:scale-105 shadow-2xl shadow-emerald-500/25 cursor-pointer"
            onClick={handleStart}>
              <span className="flex items-center space-x-2 text-lg font-semibold">
                <span>Start The Application</span>
                <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </span>
            </button>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto">
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <div className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent mb-2">
                  {stat.number}
                </div>
                <div className="text-gray-400 text-sm">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Features Preview */}
      <div className="relative z-10 max-w-7xl mx-auto px-6 pb-32">
        <div className="grid md:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div 
              key={index}
              className="group p-8 rounded-2xl bg-gray-800/30 backdrop-blur-lg border border-gray-700/30 hover:bg-gray-700/40 transition-all duration-500 hover:transform hover:scale-105"
            >
              <div className="w-16 h-16 bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <feature.icon className="w-8 h-8 text-emerald-400" />
              </div>
              <h3 className="text-xl font-semibold mb-4 text-white">{feature.title}</h3>
              <p className="text-gray-400 leading-relaxed">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>


      {/* Scroll Indicator */}
      <div className="fixed bottom-8 left-1/2 transform -translate-x-1/2 z-50 animate-bounce">
        <div className="w-6 h-10 border-2 border-gray-600 rounded-full flex justify-center">
          <div className="w-1 h-3 bg-gradient-to-b from-emerald-400 to-cyan-400 rounded-full mt-2 animate-pulse"></div>
        </div>
      </div>
    </div>
  );
};

export default NutritionHero;