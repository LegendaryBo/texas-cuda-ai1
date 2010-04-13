package pl.wroc.uni.ii.evolution.distribution.workers.evEvolutionApplet;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;

import pl.wroc.uni.ii.evolution.chart.EvObjectiveFunctionValueMaxAvgMinChart;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;

@SuppressWarnings("serial")
public class EvChartPanel extends ChartPanel {

  public EvChartPanel() {
    // empty chart
    super(EvObjectiveFunctionValueMaxAvgMinChart.createJFreeChart(
        new EvStatistic[0], false));
  }


  public void setStats(EvPersistentStatisticStorage storage) {
    JFreeChart chart =
        EvObjectiveFunctionValueMaxAvgMinChart.createJFreeChart(storage
            .getStatistics(), false);
    this.setChart(chart);
  }

}
