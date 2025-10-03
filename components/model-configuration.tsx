"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Settings, Play } from "lucide-react"

interface ModelConfigurationProps {
  data: any
  onNext: () => void
}

export function ModelConfiguration({ data, onNext }: ModelConfigurationProps) {
  const [config, setConfig] = useState({
    p: 1,
    q: 1,
    distribution: "normal",
    windowSize: 252,
    forecastHorizon: 20,
  })
  const [loading, setLoading] = useState(false)

  const handleRunModel = async () => {
    setLoading(true)
    try {
      const response = await fetch("/api/run-garch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      })

      if (response.ok) {
        onNext()
      }
    } catch (error) {
      console.error("[v0] Error running model:", error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-3">
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5 text-primary" />
            GARCH Model Configuration
          </CardTitle>
          <CardDescription>Configure model parameters and forecasting settings</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="p-order">GARCH p (lag order)</Label>
              <Input
                id="p-order"
                type="number"
                min="1"
                max="5"
                value={config.p}
                onChange={(e) => setConfig({ ...config, p: Number.parseInt(e.target.value) })}
              />
              <p className="text-xs text-muted-foreground">Number of lagged variance terms</p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="q-order">GARCH q (lag order)</Label>
              <Input
                id="q-order"
                type="number"
                min="1"
                max="5"
                value={config.q}
                onChange={(e) => setConfig({ ...config, q: Number.parseInt(e.target.value) })}
              />
              <p className="text-xs text-muted-foreground">Number of lagged residual terms</p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="distribution">Error Distribution</Label>
              <Select
                value={config.distribution}
                onValueChange={(value) => setConfig({ ...config, distribution: value })}
              >
                <SelectTrigger id="distribution">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="normal">Normal</SelectItem>
                  <SelectItem value="t">Student-t</SelectItem>
                  <SelectItem value="skewt">Skewed Student-t</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="window">Rolling Window Size</Label>
              <Input
                id="window"
                type="number"
                min="50"
                max="1000"
                value={config.windowSize}
                onChange={(e) => setConfig({ ...config, windowSize: Number.parseInt(e.target.value) })}
              />
              <p className="text-xs text-muted-foreground">Number of observations for estimation</p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="horizon">Forecast Horizon</Label>
              <Input
                id="horizon"
                type="number"
                min="1"
                max="100"
                value={config.forecastHorizon}
                onChange={(e) => setConfig({ ...config, forecastHorizon: Number.parseInt(e.target.value) })}
              />
              <p className="text-xs text-muted-foreground">Number of periods to forecast</p>
            </div>
          </div>

          <Button onClick={handleRunModel} disabled={loading || !data} className="w-full gap-2">
            <Play className="h-4 w-4" />
            {loading ? "Running Model..." : "Run GARCH Model"}
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Model Information</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-3 text-sm">
            <div>
              <p className="font-medium text-foreground">GARCH(p,q)</p>
              <p className="text-muted-foreground">
                Generalized Autoregressive Conditional Heteroskedasticity model for volatility forecasting
              </p>
            </div>
            <div>
              <p className="font-medium text-foreground">Current Configuration</p>
              <div className="mt-2 space-y-1 rounded-lg bg-muted/50 p-3 font-mono text-xs">
                <p>
                  Model: GARCH({config.p},{config.q})
                </p>
                <p>Distribution: {config.distribution}</p>
                <p>Window: {config.windowSize} days</p>
                <p>Horizon: {config.forecastHorizon} periods</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
