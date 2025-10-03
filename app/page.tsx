"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DataUpload } from "@/components/data-upload"
import { ModelConfiguration } from "@/components/model-configuration"
import { ForecastResults } from "@/components/forecast-results"
import { ModelDiagnostics } from "@/components/model-diagnostics"
import { AdvancedFeatures } from "@/components/advanced-features"
import { Activity, TrendingUp, BarChart3, Settings } from "lucide-react"

export default function VolatilityDashboard() {
  const [activeTab, setActiveTab] = useState("upload")
  const [analysisData, setAnalysisData] = useState<any>(null)

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
                <Activity className="h-6 w-6 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-foreground">Volatility Forecasting Engine</h1>
                <p className="text-sm text-muted-foreground">Advanced GARCH modeling and risk analytics</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <div className="rounded-lg bg-accent/10 px-3 py-1.5">
                <span className="text-sm font-medium text-accent">GARCH(1,1)</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-5 lg:w-auto lg:inline-grid">
            <TabsTrigger value="upload" className="gap-2">
              <BarChart3 className="h-4 w-4" />
              Data
            </TabsTrigger>
            <TabsTrigger value="configure" className="gap-2">
              <Settings className="h-4 w-4" />
              Configure
            </TabsTrigger>
            <TabsTrigger value="results" className="gap-2">
              <TrendingUp className="h-4 w-4" />
              Forecast
            </TabsTrigger>
            <TabsTrigger value="diagnostics" className="gap-2">
              <Activity className="h-4 w-4" />
              Diagnostics
            </TabsTrigger>
            <TabsTrigger value="advanced" className="gap-2">
              <Activity className="h-4 w-4" />
              Advanced
            </TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="space-y-6">
            <DataUpload onDataProcessed={setAnalysisData} onNext={() => setActiveTab("configure")} />
          </TabsContent>

          <TabsContent value="configure" className="space-y-6">
            <ModelConfiguration data={analysisData} onNext={() => setActiveTab("results")} />
          </TabsContent>

          <TabsContent value="results" className="space-y-6">
            <ForecastResults data={analysisData} />
          </TabsContent>

          <TabsContent value="diagnostics" className="space-y-6">
            <ModelDiagnostics data={analysisData} />
          </TabsContent>

          <TabsContent value="advanced" className="space-y-6">
            <AdvancedFeatures data={analysisData} />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}
