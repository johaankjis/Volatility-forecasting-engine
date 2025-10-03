"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Activity, Zap, Settings2 } from "lucide-react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts"

interface AdvancedFeaturesProps {
  data: any
}

export function AdvancedFeatures({ data }: AdvancedFeaturesProps) {
  const [loading, setLoading] = useState(false)

  // Mock data
  const kalmanData = Array.from({ length: 100 }, (_, i) => ({
    period: i,
    observed: 0.02 + Math.random() * 0.03,
    filtered: 0.025 + Math.random() * 0.025,
  }))

  const varData = [
    { confidence: "95%", var: 0.0234, cvar: 0.0312 },
    { confidence: "99%", var: 0.0345, cvar: 0.0456 },
    { confidence: "99.9%", var: 0.0456, cvar: 0.0589 },
  ]

  return (
    <div className="space-y-6">
      <Tabs defaultValue="kalman" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="kalman">Kalman Filter</TabsTrigger>
          <TabsTrigger value="montecarlo">Monte Carlo</TabsTrigger>
          <TabsTrigger value="optimization">Optimization</TabsTrigger>
        </TabsList>

        <TabsContent value="kalman" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5 text-primary" />
                Kalman Filter State-Space Model
              </CardTitle>
              <CardDescription>Latent volatility estimation with maximum likelihood</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={kalmanData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis
                    dataKey="period"
                    stroke="hsl(var(--muted-foreground))"
                    tick={{ fill: "hsl(var(--muted-foreground))" }}
                  />
                  <YAxis
                    stroke="hsl(var(--muted-foreground))"
                    tick={{ fill: "hsl(var(--muted-foreground))" }}
                    tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--popover))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                      color: "hsl(var(--popover-foreground))",
                    }}
                    formatter={(value: any) => `${(value * 100).toFixed(2)}%`}
                  />
                  <Line
                    type="monotone"
                    dataKey="observed"
                    stroke="hsl(var(--chart-2))"
                    strokeWidth={1}
                    dot={false}
                    name="Observed"
                  />
                  <Line
                    type="monotone"
                    dataKey="filtered"
                    stroke="hsl(var(--chart-1))"
                    strokeWidth={2}
                    dot={false}
                    name="Filtered (Kalman)"
                  />
                </LineChart>
              </ResponsiveContainer>

              <div className="grid gap-4 md:grid-cols-3">
                <div className="rounded-lg border border-border bg-card p-4">
                  <p className="text-sm text-muted-foreground">Log-Likelihood</p>
                  <p className="text-xl font-semibold text-foreground">-1,234.56</p>
                </div>
                <div className="rounded-lg border border-border bg-card p-4">
                  <p className="text-sm text-muted-foreground">AIC</p>
                  <p className="text-xl font-semibold text-foreground">2,475.12</p>
                </div>
                <div className="rounded-lg border border-border bg-card p-4">
                  <p className="text-sm text-muted-foreground">BIC</p>
                  <p className="text-xl font-semibold text-foreground">2,489.34</p>
                </div>
              </div>

              <Button onClick={() => setLoading(true)} disabled={loading} className="w-full">
                {loading ? "Running Kalman Filter..." : "Run Kalman Filter"}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="montecarlo" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5 text-accent" />
                Monte Carlo Stress Testing
              </CardTitle>
              <CardDescription>VaR and CVaR estimation with fat-tailed distributions</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <p className="text-sm font-medium text-foreground">Simulation Parameters</p>
                  <div className="rounded-lg border border-border bg-muted/30 p-3 font-mono text-xs space-y-1">
                    <p>Simulations: 10,000</p>
                    <p>Distribution: Student-t (df=5)</p>
                    <p>Horizon: 10 days</p>
                    <p>Portfolio Value: $1,000,000</p>
                  </div>
                </div>

                <div className="space-y-2">
                  <p className="text-sm font-medium text-foreground">Risk Metrics</p>
                  <ResponsiveContainer width="100%" height={150}>
                    <BarChart data={varData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis
                        type="number"
                        stroke="hsl(var(--muted-foreground))"
                        tick={{ fill: "hsl(var(--muted-foreground))" }}
                        tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
                      />
                      <YAxis
                        type="category"
                        dataKey="confidence"
                        stroke="hsl(var(--muted-foreground))"
                        tick={{ fill: "hsl(var(--muted-foreground))" }}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--popover))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "8px",
                          color: "hsl(var(--popover-foreground))",
                        }}
                        formatter={(value: any) => `${(value * 100).toFixed(2)}%`}
                      />
                      <Bar dataKey="var" fill="hsl(var(--chart-1))" name="VaR" />
                      <Bar dataKey="cvar" fill="hsl(var(--chart-2))" name="CVaR" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="grid gap-4 md:grid-cols-3">
                <div className="rounded-lg border border-border bg-card p-4">
                  <p className="text-sm text-muted-foreground">VaR (95%)</p>
                  <p className="text-xl font-semibold text-destructive">-$23,400</p>
                </div>
                <div className="rounded-lg border border-border bg-card p-4">
                  <p className="text-sm text-muted-foreground">CVaR (95%)</p>
                  <p className="text-xl font-semibold text-destructive">-$31,200</p>
                </div>
                <div className="rounded-lg border border-border bg-card p-4">
                  <p className="text-sm text-muted-foreground">Max Drawdown</p>
                  <p className="text-xl font-semibold text-destructive">-$45,600</p>
                </div>
              </div>

              <Button onClick={() => setLoading(true)} disabled={loading} className="w-full">
                {loading ? "Running Simulation..." : "Run Monte Carlo Simulation"}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="optimization" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings2 className="h-5 w-5 text-primary" />
                Hyperparameter Optimization
              </CardTitle>
              <CardDescription>Grid search and Bayesian optimization for model selection</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <div className="rounded-lg border border-border p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-foreground">Grid Search</p>
                      <p className="text-sm text-muted-foreground">Exhaustive search over parameter grid</p>
                    </div>
                    <Button variant="outline">Configure</Button>
                  </div>
                </div>

                <div className="rounded-lg border border-border p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-foreground">Bayesian Optimization</p>
                      <p className="text-sm text-muted-foreground">Efficient search using Gaussian processes</p>
                    </div>
                    <Button variant="outline">Configure</Button>
                  </div>
                </div>
              </div>

              <div className="rounded-lg border border-border bg-muted/30 p-4">
                <p className="mb-2 text-sm font-medium text-foreground">Best Model Found</p>
                <div className="space-y-1 font-mono text-xs">
                  <p>Model: GARCH(1,1)</p>
                  <p>Distribution: Student-t</p>
                  <p>Log-Likelihood: -1,234.56</p>
                  <p>AIC: 2,475.12</p>
                </div>
              </div>

              <Button onClick={() => setLoading(true)} disabled={loading} className="w-full">
                {loading ? "Optimizing..." : "Start Optimization"}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
