package pl.wroc.uni.ii.evolution.engine.operators.general.display;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * An operator to gather statistics. It keeps track of max, min, avg and
 * standard deviation of the goal function of individuals in population. It can
 * produce an input dat file for gnuplot.
 * 
 * @author Kamil Dworakowski
 * @param <T>
 */
public class EvGatherStatistics<T extends EvIndividual> implements
    EvOperator<T> {

  private static class IterationInfo {
    double max, min, mean, stdev;


    IterationInfo(double max, double min, double mean, double std) {
      super();
      this.max = max;
      this.min = min;
      this.mean = mean;
      this.stdev = std;
    }
  }

  private List<IterationInfo> infos = new ArrayList<IterationInfo>();

  int iteration_number = 1;


  public EvPopulation<T> apply(EvPopulation<T> population) {
    double avg = 0.0;
    for (T i : population) {
      avg += i.getObjectiveFunctionValue();
    }

    avg /= population.size();

    double stddev = 0.0;
    for (T i : population) {
      stddev +=
          (i.getObjectiveFunctionValue() - avg)
              * (i.getObjectiveFunctionValue() - avg);
    }

    stddev = Math.sqrt(stddev / population.size());

    double min = Double.MAX_VALUE, max = Double.MIN_VALUE;
    for (T i : population) {
      if (i.getObjectiveFunctionValue() > max)
        max = i.getObjectiveFunctionValue();
      if (i.getObjectiveFunctionValue() < min)
        min = i.getObjectiveFunctionValue();
    }

    infos.add(new IterationInfo(max, min, avg, stddev));

    return population;
  }


  public double lastMean() {
    if (infos.size() > 0)
      return infos.get(infos.size() - 1).mean;
    else
      return 0;
  }


  public double lastMax() {
    if (infos.size() > 0)
      return infos.get(infos.size() - 1).max;
    else
      return 0;
  }


  public double lastMin() {
    if (infos.size() > 0)
      return infos.get(infos.size() - 1).min;
    else
      return 0;
  }


  public double lastStD() {
    if (infos.size() > 0)
      return infos.get(infos.size() - 1).stdev;
    else
      return 0;
  }


  private void write(Writer writer, double val) throws IOException {
    writer.write(Double.toString(val));
    writer.write('\t');
  }


  public void writeForGnuplot(Writer writer) throws IOException {
    for (int i = 0; i < infos.size(); i++) {
      IterationInfo info = infos.get(i);
      write(writer, i + 1);
      write(writer, info.max);
      write(writer, info.mean);
      write(writer, info.min);
      write(writer, info.stdev);
      writer.write('\n');
    }
  }


  private String plotCurveNumber(String file, int nr, String title) {
    return "\"" + file + "\"" + "using 1:" + nr + " title '" + title
        + "' with lines";
  }


  /**
   * Does the same as writeToGnuplotFile, but it creates a command file for
   * gnuplot in addition. The commond file created has .p suffix.
   * <p>
   * Command 'load "file_name.p"' in gnuplot will invoke the script and plot the
   * data.
   * 
   * @param file_name
   * @throws IOException
   */
  public void writeToGnuplotFileWithCmd(String file_path) throws IOException {
    String file_name = new File(file_path).getName();
    writeToGnuplotFile(file_path);
    FileWriter writer = new FileWriter(file_path + ".p");
    writer.write("plot " + plotCurveNumber(file_name, 2, "max") + ", "
        + plotCurveNumber(file_name, 3, "mean") + ", "
        + plotCurveNumber(file_name, 4, "min") + ", "
        + plotCurveNumber(file_name, 5, "StD"));
    writer.close();
  }


  /**
   * Cmd file will make gnuplot create an eps file.
   * <p>
   * in windows try pgnuplot cmdfile_name
   * 
   * @param file_name
   * @throws IOException
   */
  public void writeToGnuplotFileWithCmdToEPS(String file_path)
      throws IOException {
    String file_name = new File(file_path).getName();
    writeToGnuplotFile(file_path);
    FileWriter writer = new FileWriter(file_path + ".p");
    writer.write("set term postscript eps enhanced\n");
    writer.write("set output \"" + file_name + ".eps\"\n");
    writer.write("plot " + plotCurveNumber(file_name, 2, "max") + ", "
        + plotCurveNumber(file_name, 3, "mean") + ", "
        + plotCurveNumber(file_name, 4, "min") + ", "
        + plotCurveNumber(file_name, 5, "StD"));
    writer.close();
  }


  /**
   * Cmd file will make gnuplot create an eps file.
   * <p>
   * in windows try pgnuplot cmdfile_name
   * 
   * @param minValue
   * @param file_name
   * @throws IOException
   */
  public void writeToGnuplotFileWithCmdToEPS(String file_path, double minValue,
      double maxValue) throws IOException {
    String file_name = new File(file_path).getName();
    writeToGnuplotFile(file_path);
    FileWriter writer = new FileWriter(file_path + ".p");
    writer.write("set term postscript eps enhanced\n");
    writer.write("set output \"" + file_name + ".eps\"\n");
    writer.write("plot [:] [" + minValue + ":" + maxValue + "] "
        + plotCurveNumber(file_name, 2, "max") + ", "
        + plotCurveNumber(file_name, 3, "mean") + ", "
        + plotCurveNumber(file_name, 4, "min") + ", "
        + plotCurveNumber(file_name, 5, "StD"));
    writer.close();
  }


  /**
   * Writes the data held to a file in format acceptable by GNUPlot. The data is
   * printed in four columns: max, mean, min, and st. deviation respectively.
   * 
   * @param file_name file_name where to write the data to plot
   * @throws IOException
   */
  public void writeToGnuplotFile(String file_name) throws IOException {
    FileWriter writer = new FileWriter(file_name);
    writeForGnuplot(writer);
    writer.close();
  }
}
