"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Activity, CheckCircle2, XCircle } from "lucide-react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
} from "recharts"

interface ModelDiagnosticsProps {
  data: any
}

export function ModelDiagnostics({ data }: ModelDiagnosticsProps) {
  // Mock diagnostic data
  const residualData = Array.from({ length: 100 }, (_, i) => ({
    index: i,
    residual: (Math.random() - 0.5) * 0.1,
    squared: Math.random() * 0.01,
  }))

  const acfData = Array.from({ length: 20 }, (_, i) => ({
    lag: i + 1,
    acf: Math.random() * 0.3 - 0.15,
  }))

  const diagnosticTests = [
    { name: "Ljung-Box Test", statistic: 18.45, pValue: 0.234, passed: true },
    { name: "ARCH-LM Test", statistic: 2.34, pValue: 0.673, passed: true },
    { name: "Jarque-Bera Test", statistic: 5.67, pValue: 0.059, passed: true },
  ]

  const modelParams = {
    omega: 0.000012,
    alpha: 0.085,
    beta: 0.905,
    persistence: 0.99,
    halfLife: 68.5,
    unconditionalVol: 0.0234,
  }

  return (
    <div className="space-y-6">
      {/* Model Parameters */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Model Persistence</CardDescription>
            <CardTitle className="text-2xl text-foreground">{modelParams.persistence.toFixed(3)}</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">α + β = {modelParams.persistence.toFixed(3)}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Half-Life of Shocks</CardDescription>
            <CardTitle className="text-2xl text-foreground">{modelParams.halfLife.toFixed(1)}</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">Days for shock to decay 50%</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Unconditional Vol</CardDescription>
            <CardTitle className="text-2xl text-foreground">
              {(modelParams.unconditionalVol * 100).toFixed(2)}%
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">Long-run average volatility</p>
          </CardContent>
        </Card>
      </div>

      {/* Parameter Estimates */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" />
            GARCH Parameter Estimates
          </CardTitle>
          <CardDescription>Maximum likelihood estimates with standard errors</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="grid grid-cols-3 gap-4 rounded-lg border border-border bg-muted/30 p-4">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Parameter</p>
              </div>
              <div>
                <p className="text-sm font-medium text-muted-foreground">Estimate</p>
              </div>
              <div>
                <p className="text-sm font-medium text-muted-foreground">Std Error</p>
              </div>
            </div>
            {[
              { param: "ω (omega)", value: modelParams.omega, se: 0.000003 },
              { param: "α (alpha)", value: modelParams.alpha, se: 0.012 },
              { param: "β (beta)", value: modelParams.beta, se: 0.015 },
            ].map((row) => (
              <div key={row.param} className="grid grid-cols-3 gap-4 rounded-lg border border-border p-4">
                <div>
                  <p className="font-mono text-sm text-foreground">{row.param}</p>
                </div>
                <div>
                  <p className="font-mono text-sm text-foreground">{row.value.toFixed(6)}</p>
                </div>
                <div>
                  <p className="font-mono text-sm text-muted-foreground">{row.se.toFixed(6)}</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Diagnostic Tests */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-accent" />
            Diagnostic Tests
          </CardTitle>
          <CardDescription>Statistical tests for model adequacy</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {diagnosticTests.map((test) => (
              <div key={test.name} className="flex items-center justify-between rounded-lg border border-border p-4">
                <div className="flex items-center gap-3">
                  {test.passed ? (
                    <CheckCircle2 className="h-5 w-5 text-chart-3" />
                  ) : (
                    <XCircle className="h-5 w-5 text-destructive" />
                  )}
                  <div>
                    <p className="font-medium text-foreground">{test.name}</p>
                    <p className="text-sm text-muted-foreground">
                      Statistic: {test.statistic.toFixed(2)} | p-value: {test.pValue.toFixed(3)}
                    </p>
                  </div>
                </div>
                <div
                  className={`rounded-full px-3 py-1 text-xs font-medium ${
                    test.passed ? "bg-chart-3/10 text-chart-3" : "bg-destructive/10 text-destructive"
                  }`}
                >
                  {test.passed ? "Pass" : "Fail"}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Residual Analysis */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Standardized Residuals</CardTitle>
            <CardDescription>Time series plot of model residuals</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="index"
                  stroke="hsl(var(--muted-foreground))"
                  tick={{ fill: "hsl(var(--muted-foreground))" }}
                />
                <YAxis stroke="hsl(var(--muted-foreground))" tick={{ fill: "hsl(var(--muted-foreground))" }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--popover))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                    color: "hsl(var(--popover-foreground))",
                  }}
                />
                <Scatter data={residualData} fill="hsl(var(--chart-1))" />
              </ScatterChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>ACF of Squared Residuals</CardTitle>
            <CardDescription>Test for remaining ARCH effects</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={acfData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="lag"
                  stroke="hsl(var(--muted-foreground))"
                  tick={{ fill: "hsl(var(--muted-foreground))" }}
                />
                <YAxis stroke="hsl(var(--muted-foreground))" tick={{ fill: "hsl(var(--muted-foreground))" }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--popover))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                    color: "hsl(var(--popover-foreground))",
                  }}
                />
                <Bar dataKey="acf" fill="hsl(var(--chart-2))" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
