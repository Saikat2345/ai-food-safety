import { useState } from 'react'
import NutritionAnalyzer from './Components/NutritionAnalyzer'
import NutritionHero from './Components/NeutriHero'
import { Routes,Route } from 'react-router-dom'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      {/* <NutritionAnalyzer/> */}
      <Routes>
        <Route path="/" element={<NutritionHero />} />
        <Route path="/analyze" element={<NutritionAnalyzer />} />
      </Routes>
    </>
  )
}

export default App
