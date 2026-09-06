import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'
import type { ComponentIntelligence } from '../types/api'
import { num } from '../utils/format'

export function TrajectoryChart({ data }: { data: ComponentIntelligence }) {
  const ref = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!ref.current) return
    const chart = echarts.init(ref.current)
    const hist = data.historical_trajectory.filter(p => p.value != null)
    const f = data.forecast
    const observed = hist.map(p => [p.time_h, p.value as number])
    const origin = f.origin_h ?? (hist.length ? hist[hist.length - 1].time_h : 0)
    const lastValue = hist.length ? hist[hist.length - 1].value as number : f.prediction
    const horizon = f.horizon_h ?? 168
    const prediction = f.prediction
    const lower = f.lower ?? prediction
    const upper = f.upper ?? prediction
    const limitEntry = Object.entries(data.engineering_limits).find(([k]) => /upper|spec_upper|limit_upper/i.test(k))
    const limit = limitEntry?.[1] ?? null

    const forecast = prediction == null ? [] : [[origin, lastValue], [horizon, prediction]]
    const bandLower = prediction == null ? [] : [[origin, lower], [horizon, lower]]
    const bandWidth = prediction == null ? [] : [[origin, Math.max(0, (upper ?? prediction) - (lower ?? prediction))], [horizon, Math.max(0, (upper ?? prediction) - (lower ?? prediction))]]

    const option: echarts.EChartsOption = {
      backgroundColor: 'transparent',
      animation: false,
      grid: { left: 62, right: 28, top: 28, bottom: 52 },
      tooltip: { trigger: 'axis', backgroundColor: '#171311', borderColor: '#5b3e2f', textStyle: { color: '#efe2d6', fontSize: 10 } },
      xAxis: { type: 'value', name: 'BURN-IN TIME (h)', nameTextStyle: { color: '#8f7d72', fontSize: 9 }, axisLabel: { color: '#8f7d72', fontSize: 9 }, splitLine: { lineStyle: { color: '#2d241f' } } },
      yAxis: { type: 'value', name: data.current_measurements.unit ? String(data.current_measurements.unit) : 'value', nameTextStyle: { color: '#8f7d72', fontSize: 9 }, axisLabel: { color: '#8f7d72', fontSize: 9 }, splitLine: { lineStyle: { color: '#2d241f' } } },
      series: [
        { name: 'Observed', type: 'line', data: observed, symbol: 'circle', symbolSize: 5, lineStyle: { color: '#e8dbd0', width: 2 }, itemStyle: { color: '#e8dbd0' } },
        { name: 'Forecast', type: 'line', data: forecast, symbol: 'circle', symbolSize: 5, lineStyle: { color: '#cc7a55', width: 2.2, type: 'dashed' }, itemStyle: { color: '#cc7a55' } },
        { name: 'Lower bound', type: 'line', data: bandLower, stack: 'confidence-band', symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { opacity: 0 } },
        { name: 'Uncertainty', type: 'line', data: bandWidth, stack: 'confidence-band', symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { color: 'rgba(196,122,74,.14)' } },
        { name: 'Upper bound', type: 'line', data: prediction == null ? [] : [[origin, upper], [horizon, upper]], symbol: 'none', lineStyle: { color: '#a9573e', width: 1, type: 'dotted' } },
        { name: 'Limit', type: 'line', data: limit == null ? [] : [[0, limit], [horizon, limit]], symbol: 'none', lineStyle: { color: '#c99a55', width: 1.4 }, markLine: limit == null ? undefined : { silent: true, data: [{ yAxis: limit }], label: { formatter: `LIMIT ${num(limit, 2)}`, color: '#c99a55', fontSize: 9 } } },
      ],
    }
    chart.setOption(option)
    const ro = new ResizeObserver(() => chart.resize())
    ro.observe(ref.current)
    return () => { ro.disconnect(); chart.dispose() }
  }, [data])

  return <div ref={ref} className="chart" />
}
