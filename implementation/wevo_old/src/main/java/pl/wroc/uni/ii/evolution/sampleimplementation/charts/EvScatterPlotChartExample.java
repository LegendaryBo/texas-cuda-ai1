package pl.wroc.uni.ii.evolution.sampleimplementation.charts;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

import org.jfree.chart.JFreeChart;

import pl.wroc.uni.ii.evolution.chart.EvChartTools;
import pl.wroc.uni.ii.evolution.chart.EvScatterPlotChart;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvApplyOnSelectionComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvIterateCompositon;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvReplacementComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvTwoOperatorsComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvTwoSelectionComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvRandomSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentSimpleStorage;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.scatterplot.EvRealVectorScatterPlotGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector.EvRealVectorAverageCrossover;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector.EvRealVectorStandardDeviationMutation;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvRealVectorSpace;

/**
 * Simple example of usage of Scatter plot charts. It runs simple algorithm on
 * small real vector problem and shows progress in aplet in 4 different stages
 * 
 * @author Kacper Gorski (admin@34all.org)
 */
public class EvScatterPlotChartExample {

  /**
   * @param str - application doesn't use parameters
   */
  public static void main(final String[] str) {

    EvAlgorithm<EvRealVectorIndividual> genericEA =
        new EvAlgorithm<EvRealVectorIndividual>(1000);

    genericEA.setSolutionSpace(new EvRealVectorSpace(
        new EvRealOneMax<EvRealVectorIndividual>(), 3));
    genericEA.setObjectiveFunction(new EvRealOneMax<EvRealVectorIndividual>());

    EvPersistentSimpleStorage storage = new EvPersistentSimpleStorage();

    EvRealVectorScatterPlotGatherer gatherer =
        new EvRealVectorScatterPlotGatherer(storage);

    genericEA.addOperator(gatherer);

    genericEA
        .addOperatorToEnd(new EvReplacementComposition<EvRealVectorIndividual>(
            new EvIterateCompositon<EvRealVectorIndividual>(
                new EvApplyOnSelectionComposition<EvRealVectorIndividual>(
                    new EvTwoSelectionComposition<EvRealVectorIndividual>(
                        new EvRandomSelection<EvRealVectorIndividual>(16, false),
                        new EvKBestSelection<EvRealVectorIndividual>(4)),
                    new EvTwoOperatorsComposition<EvRealVectorIndividual>(
                        new EvRealVectorStandardDeviationMutation(0.01d),
                        new EvRealVectorAverageCrossover(4)))),
            new EvBestFromUnionReplacement<EvRealVectorIndividual>()));

    genericEA
        .setTerminationCondition(new EvMaxIteration<EvRealVectorIndividual>(12));

    genericEA.addOperatorToEnd(
        new EvRealtimeToPrintStreamStatistics(
            System.out));
    
    genericEA.init();

    genericEA.run();

    int[][] pair_genes = new int[3][2];
    pair_genes[0][0] = 0;
    pair_genes[0][1] = 1;
    pair_genes[1][0] = 0;
    pair_genes[1][1] = 2;
    pair_genes[2][0] = 1;
    pair_genes[2][1] = 2;

    buildGraphics(storage.getStatistics(), pair_genes);

  }


  /**
   * Builds charts on graphics interface.
   * 
   * @param statistics -
   * @param pair_genes -
   */
  private static void buildGraphics(final EvStatistic[] statistics,
      final int[][] pair_genes) {

    JFreeChart[] chart =
        EvScatterPlotChart.createJFreeChart(statistics, pair_genes, 1);

    JPanel panel1 = EvChartTools.freeChartToJPanel(chart[0]);
    JPanel panel2 = EvChartTools.freeChartToJPanel(chart[1]);
    JPanel panel3 = EvChartTools.freeChartToJPanel(chart[2]);
    JLabel label = new JLabel("Iteration 1");

    JFrame f = new JFrame();
    f.setLayout(null);

    panel1.setBounds(300, 0, 300, 200);
    f.add(panel1);
    panel2.setBounds(0, 200, 300, 200);
    f.add(panel2);
    panel3.setBounds(300, 200, 300, 200);
    f.add(panel3);
    label.setBounds(130, 20, 200, 90);
    f.add(label);

    chart = EvScatterPlotChart.createJFreeChart(statistics, pair_genes, 3);

    panel1 = EvChartTools.freeChartToJPanel(chart[0]);
    panel2 = EvChartTools.freeChartToJPanel(chart[1]);
    panel3 = EvChartTools.freeChartToJPanel(chart[2]);
    label = new JLabel("Iteration 3");

    panel1.setBounds(900, 0, 300, 200);
    f.add(panel1);
    panel2.setBounds(600, 200, 300, 200);
    f.add(panel2);
    panel3.setBounds(900, 200, 300, 200);
    f.add(panel3);
    label.setBounds(730, 20, 200, 90);
    f.add(label);

    chart = EvScatterPlotChart.createJFreeChart(statistics, pair_genes, 6);

    panel1 = EvChartTools.freeChartToJPanel(chart[0]);
    panel2 = EvChartTools.freeChartToJPanel(chart[1]);
    panel3 = EvChartTools.freeChartToJPanel(chart[2]);
    label = new JLabel("Iteration 6");

    panel1.setBounds(300, 400, 300, 200);
    f.add(panel1);
    panel2.setBounds(0, 600, 300, 200);
    f.add(panel2);
    panel3.setBounds(0, 600, 300, 200);
    f.add(panel3);
    label.setBounds(130, 420, 200, 90);
    f.add(label);

    chart = EvScatterPlotChart.createJFreeChart(statistics, pair_genes, 8);

    panel1 = EvChartTools.freeChartToJPanel(chart[0]);
    panel2 = EvChartTools.freeChartToJPanel(chart[1]);
    panel3 = EvChartTools.freeChartToJPanel(chart[2]);
    label = new JLabel("Iteration 8");

    panel1.setBounds(900, 400, 300, 200);
    f.add(panel1);
    panel2.setBounds(600, 600, 300, 200);
    f.add(panel2);
    panel3.setBounds(900, 600, 300, 200);
    f.add(panel3);
    label.setBounds(730, 420, 200, 90);
    f.add(label);

    f.pack();
    f.setSize(1200, 800); // add 20, seems enough for the Frame title,
    f.show();

  }


  /**
   * Disabling constructor.
   */
  protected EvScatterPlotChartExample() {
    throw new UnsupportedOperationException();
  }

}
