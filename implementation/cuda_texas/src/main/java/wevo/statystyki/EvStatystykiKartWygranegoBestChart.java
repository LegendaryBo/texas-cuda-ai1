package wevo.statystyki;

import java.awt.Color;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;

public class EvStatystykiKartWygranegoBestChart {

  
  /**
   * @param stats
   * @param logarithmic
   * @param genes_indexes
   */
  public static JFreeChart createJFreeChart(final EvStatistic[] stats) {
    final JFreeChart chart =
        ChartFactory.createXYLineChart("Srednia wygrana/przegrana (najlepszy osobnik)", "Iteration",
            "Wysokosc wygraneh/przegranej", getDataSet(stats),
            PlotOrientation.VERTICAL, true, false, false);

    // get a reference to the plot for further customization...
    final XYPlot plot = chart.getXYPlot();
    plot.setBackgroundPaint(Color.WHITE);
    plot.setRangeGridlinePaint(Color.GRAY);

    ValueAxis axis = plot.getDomainAxis();
    axis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
    return chart;

  }




  /**
   * It constructs XYDataset object.
   * 
   * @param storage - EvAvgGenesStatisticStorage object in which data is stored
   * @param genes_indexes - index of genes to be displayed on chart
   * @param starting_iteration - first iteration shown on chart
   * @param ending_iteration - last iteration shown on chart
   * @return XYDataset object which can be used to generate statistics by their
   *         classes
   */
  public static XYDataset getDataSet(final EvStatistic[] stats) {

    // data containing information of single gene
    XYSeries[] series = null;

    // data containing all the info on the charts
    XYSeriesCollection dataset = new XYSeriesCollection();

    if (stats != null) {

      for (EvStatistic stat : stats) {
        int[] genes_value = ((EvStatystykiKartWygranegoStatistic) stat).statystykiKartBest;

          if (series == null) {
            series = new XYSeries[7];
            series[0] = new XYSeries("Smiec (najlepszy os.)");
            dataset.addSeries(series[0]);
            series[1] = new XYSeries("Para (najlepszy os.)");
            dataset.addSeries(series[1]);
            series[2] = new XYSeries("2 Pary (najlepszy os.)");
            dataset.addSeries(series[2]);            
            series[3] = new XYSeries("Trojka (najlepszy os.)");
            dataset.addSeries(series[3]);       
            series[4] = new XYSeries("Street (najlepszy os.)");
            dataset.addSeries(series[4]);    
            series[5] = new XYSeries("Kolor (najlepszy os.)");
            dataset.addSeries(series[5]);   
            series[6] = new XYSeries("Full, Kareta, Poker  (najlepszy os.)");
            dataset.addSeries(series[6]);   
          } 
     
        for (int j = 0; j < genes_value.length; j++) {
          series[j].add(stat.getIteration(), genes_value[j]);
        }

      }

    }
    return dataset;
  }
    
  
  
}
