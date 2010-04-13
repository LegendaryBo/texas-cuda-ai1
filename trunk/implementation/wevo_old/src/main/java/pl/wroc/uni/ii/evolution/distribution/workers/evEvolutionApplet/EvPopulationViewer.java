package pl.wroc.uni.ii.evolution.distribution.workers.evEvolutionApplet;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Simple component to view detailed information avout population
 * 
 * @author Kacper Gorski
 */
@SuppressWarnings("serial")
public class EvPopulationViewer extends JFrame {

  private JTextArea text_area = new JTextArea();

  private JScrollPane scroll = new JScrollPane(text_area);

  private JLabel title_label = new JLabel();


  public EvPopulationViewer(String title) {
    setLayout(null);
    scroll.setBounds(10, 30, 600, 500);
    add(scroll);
    scroll.setPreferredSize(new java.awt.Dimension(600, 500));
    title_label.setBounds(10, 10, 400, 20);
    add(title_label);
    title_label.setText("Population viewer - " + title);
  }


  /**
   * Contruct new component and display details of given population
   * 
   * @param population to be displaed
   * @param title of this component
   */
  public EvPopulationViewer(EvPopulation population, String title) {
    this(title);
    setPopulation(population);
  }


  /**
   * Show details of given population and displays it on the component
   * 
   * @param population
   */
  public void setPopulation(EvPopulation population) {
    String txt = new String();
    if (population == null)
      txt = "null population";
    else {
      txt = "Population size: " + population.size();
      if (population.size() != 0) {
        txt +=
            "\nPopulation of: "
                + population.getBestResult().getClass().getCanonicalName();
        txt +=
            "\nBest Individual obj. fun. value: "
                + population.getBestResult().getObjectiveFunctionValue()
                + " ,string representation: " + population.getBestResult();
        txt +=
            "\nWorst Individualo obj. fun. value: "
                + population.getWorstResult().getObjectiveFunctionValue()
                + " ,string representation: " + population.getWorstResult();
        txt += "\n\nPopulation individuals:";
        for (int i = 0; i < population.size(); i++) {
          EvIndividual current_ind = (EvIndividual) population.get(i);
          txt +=
              "\nIndividual nr. " + i + " - Obj. fun. value:"
                  + current_ind.getObjectiveFunctionValue()
                  + " ,string representation: " + current_ind;
        }

      }
    }
    text_area.setText(txt);
  }

}
