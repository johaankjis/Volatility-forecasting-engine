"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Upload, FileText, CheckCircle2, AlertCircle, TrendingUp } from "lucide-react"

interface DataUploadProps {
  onDataProcessed: (data: any) => void
  onNext: () => void
}

export function DataUpload({ onDataProcessed, onNext }: DataUploadProps) {
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setError(null)
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append("file", file)

      const response = await fetch("/api/preprocess", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) throw new Error("Failed to process data")

      const data = await response.json()
      setResults(data)
      onDataProcessed(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {/* Upload Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5 text-primary" />
            Upload Time Series Data
          </CardTitle>
          <CardDescription>Upload CSV file with date and price columns for volatility analysis</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="file-upload">Select CSV File</Label>
            <Input id="file-upload" type="file" accept=".csv" onChange={handleFileChange} disabled={loading} />
          </div>

          {file && (
            <div className="flex items-center gap-2 rounded-lg border border-border bg-muted/50 p-3">
              <FileText className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm text-foreground">{file.name}</span>
            </div>
          )}

          <Button onClick={handleUpload} disabled={!file || loading} className="w-full">
            {loading ? "Processing..." : "Process Data"}
          </Button>

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Results Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-accent" />
            Data Summary
          </CardTitle>
          <CardDescription>Preprocessing results and stationarity tests</CardDescription>
        </CardHeader>
        <CardContent>
          {!results ? (
            <div className="flex h-64 items-center justify-center text-center">
              <div className="space-y-2">
                <FileText className="mx-auto h-12 w-12 text-muted-foreground" />
                <p className="text-sm text-muted-foreground">Upload data to see preprocessing results</p>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="rounded-lg border border-border bg-card p-4">
                  <p className="text-sm text-muted-foreground">Observations</p>
                  <p className="text-2xl font-semibold text-foreground">{results.observations}</p>
                </div>
                <div className="rounded-lg border border-border bg-card p-4">
                  <p className="text-sm text-muted-foreground">Mean Return</p>
                  <p className="text-2xl font-semibold text-foreground">{results.mean_return?.toFixed(4)}%</p>
                </div>
                <div className="rounded-lg border border-border bg-card p-4">
                  <p className="text-sm text-muted-foreground">Volatility</p>
                  <p className="text-2xl font-semibold text-foreground">{results.volatility?.toFixed(4)}%</p>
                </div>
                <div className="rounded-lg border border-border bg-card p-4">
                  <p className="text-sm text-muted-foreground">ADF Test</p>
                  <div className="flex items-center gap-2">
                    {results.is_stationary ? (
                      <>
                        <CheckCircle2 className="h-5 w-5 text-chart-3" />
                        <span className="text-sm font-medium text-chart-3">Stationary</span>
                      </>
                    ) : (
                      <>
                        <AlertCircle className="h-5 w-5 text-destructive" />
                        <span className="text-sm font-medium text-destructive">Non-stationary</span>
                      </>
                    )}
                  </div>
                </div>
              </div>

              <Button onClick={onNext} className="w-full">
                Continue to Model Configuration
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
