package pl.wroc.uni.ii.evolution.sampleimplementation;

import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.ComboBoxModel;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JEditorPane;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeOnJPanelStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvBlockSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvRandomSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvRouletteSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.fitness.EvIndividualFitness;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.fitness.EvSGAFitness;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.EvKnaryVectorOnePointCrossover;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.EvKnaryVectorUniformCrossover;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorNegationMutation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * This code was edited or generated using CloudGarden's Jigloo SWT/Swing GUI
 * Builder, which is free for non-commercial use. If Jigloo is being used
 * commercially (ie, by a corporation, company or business for any purpose
 * whatever) then you should purchase a license for each developer using Jigloo.
 * Please visit www.cloudgarden.com for details. Use of Jigloo implies
 * acceptance of these licensing terms. A COMMERCIAL LICENSE HAS NOT BEEN
 * PURCHASED FOR THIS MACHINE, SO JIGLOO OR THIS CODE CANNOT BE USED LEGALLY FOR
 * ANY CORPORATE OR COMMERCIAL PURPOSE.
 */
public class EvSGAApplet extends javax.swing.JFrame implements ActionListener {
  private static final long serialVersionUID = 6766282859656662782L;

  private JPanel graph1;

  private JComboBox crossover_type;

  private JLabel population_lab;

  private JTextField population_size;

  private JLabel mut_label;

  private JTextField mutation;

  private JComboBox selection_type;

  private JLabel selection_lab;

  private JLabel crossover_lab;

  private JButton gen_problem;

  private JButton start;

  private JButton stop;

  private JLabel jLabel1;

  private JPanel settings_panel;

  private JEditorPane problem;

  private EvSGAAppletObjectiveFunction obj_func;

  private Thread computing_thread;


  /**
   * Auto-generated main method to display this JApplet inside a new JFrame.
   */
  public static void main(String[] args) {
    EvSGAApplet inst = new EvSGAApplet();
    inst.setVisible(true);
  }


  public EvSGAApplet() {
    super();
    initGUI();
  }


  private void initGUI() {
    try {
      BorderLayout thisLayout = new BorderLayout();
      getContentPane().setLayout(thisLayout);
      this.setSize(665, 570);
      {
        graph1 = new JPanel();
        getContentPane().add(graph1, BorderLayout.CENTER);
        graph1.setBounds(7, 7, 301, 210);
      }
      {
        problem = new JEditorPane();
        getContentPane().add(problem, BorderLayout.SOUTH);
        problem.setText("click 'next problem' to generate a new problem");
        problem.setBounds(14, 238, 518, 133);
        problem.setEditable(false);
        problem.setPreferredSize(new java.awt.Dimension(665, 199));
        problem.setDoubleBuffered(true);
      }
      {
        settings_panel = new JPanel();
        getContentPane().add(settings_panel, BorderLayout.EAST);
        settings_panel.setLayout(null);
        settings_panel.setPreferredSize(new java.awt.Dimension(217, 364));
        {
          gen_problem = new JButton();
          settings_panel.add(gen_problem);
          gen_problem.setText("new problem");
          gen_problem.setBounds(70, 266, 133, 28);
          gen_problem.addActionListener(this);
        }
        {
          mutation = new JTextField();
          settings_panel.add(mutation);
          mutation.setBounds(140, 49, 63, 28);
          mutation.setText("0.05");
        }
        {
          mut_label = new JLabel();
          settings_panel.add(mut_label);
          mut_label.setText("mutation probability");
          mut_label.setBounds(14, 49, 105, 28);
        }
        {
          ComboBoxModel crossover_typeModel =
              new DefaultComboBoxModel(new String[] {"Single point", "Uniform"});
          crossover_type = new JComboBox();
          settings_panel.add(crossover_type);
          crossover_type.setModel(crossover_typeModel);
          crossover_type.setBounds(98, 84, 105, 28);
        }
        {
          crossover_lab = new JLabel();
          settings_panel.add(crossover_lab);
          crossover_lab.setText("crossover");
          crossover_lab.setBounds(14, 84, 63, 28);
        }
        {
          population_size = new JTextField();
          settings_panel.add(population_size);
          population_size.setBounds(140, 119, 63, 28);
          population_size.setText("100");
        }
        {
          population_lab = new JLabel();
          settings_panel.add(population_lab);
          population_lab.setText("population size");
          population_lab.setBounds(14, 119, 77, 28);
        }
        {
          ComboBoxModel algorithmModel =
              new DefaultComboBoxModel(new String[] {"Total Random",
                  "Block Selection", "Roulette (obj. f. value)",
                  "Roulette (fitness)"});
          selection_type = new JComboBox();
          settings_panel.add(selection_type);
          selection_type.setModel(algorithmModel);
          selection_type.setBounds(105, 154, 98, 28);
        }
        {
          selection_lab = new JLabel();
          settings_panel.add(selection_lab);
          selection_lab.setText("selection");
          selection_lab.setBounds(14, 154, 63, 28);
        }
        {
          start = new JButton();
          settings_panel.add(start);
          start.setText("start");
          start.setBounds(140, 189, 63, 28);
          start.addActionListener(this);
        }
        {
          jLabel1 = new JLabel();
          settings_panel.add(jLabel1);
          jLabel1.setText("PARAMETERS");
          jLabel1.setBounds(77, 14, 77, 28);
        }
        {
          stop = new JButton();
          settings_panel.add(stop);
          stop.setText("stop");
          stop.setBounds(70, 189, 63, 28);
          stop.setEnabled(false);
          stop.addActionListener(this);
        }
      }
      createObjectiveFunciton();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }


  private void do_algorithm() {
    int pop_size = Integer.parseInt(population_size.getText());

    EvAlgorithm<EvBinaryVectorIndividual> genericEA =
        new EvAlgorithm<EvBinaryVectorIndividual>(pop_size);
    genericEA.setSolutionSpace(new EvBinaryVectorSpace(obj_func, 40));
    genericEA.setObjectiveFunction(obj_func);

    // Setting selection
    if (selection_type.getSelectedItem().equals("Total Random")) {
      genericEA
          .addOperatorToEnd(new EvRandomSelection<EvBinaryVectorIndividual>(
              pop_size, false));
      System.out.println("Total random");
    } else if (selection_type.getSelectedItem().equals("Block Selection")) {
      genericEA
          .addOperatorToEnd(new EvBlockSelection<EvBinaryVectorIndividual>(
              pop_size / 2));
    } else if (selection_type.getSelectedItem().equals(
        "Roulette (obj. f. value)")) {
      genericEA
          .addOperatorToEnd(new EvRouletteSelection<EvBinaryVectorIndividual>(
              new EvIndividualFitness<EvBinaryVectorIndividual>(), pop_size));
    } else {
      genericEA
          .addOperatorToEnd(new EvRouletteSelection<EvBinaryVectorIndividual>(
              new EvSGAFitness<EvBinaryVectorIndividual>(), pop_size));
    }

    // Setting crossover
    if (crossover_type.getSelectedItem().equals("Single point")) {
      genericEA
          .addOperatorToEnd(new EvKnaryVectorOnePointCrossover<EvBinaryVectorIndividual>());
    } else {
      genericEA
          .addOperatorToEnd(new EvKnaryVectorUniformCrossover<EvBinaryVectorIndividual>());
    }

    // Setting mutation
    genericEA.addOperatorToEnd(new EvBinaryVectorNegationMutation(Double
        .parseDouble(mutation.getText())));

    // Setting basic stats observer.
    genericEA
        .addOperatorToEnd(new EvRealtimeOnJPanelStatistics<EvBinaryVectorIndividual>(
            graph1, 0, obj_func.getSumPrices()));
    genericEA.addOperatorToEnd(new EvSGAAppletUpdateProblemDisplay(problem));
    genericEA
        .setTerminationCondition(new EvMaxIteration<EvBinaryVectorIndividual>(
            graph1.getWidth()));

    EvTask evolutionary_task = new EvTask();
    evolutionary_task.setAlgorithm(genericEA);
    evolutionary_task.run();
    evolutionary_task.printBestResult();
  }


  @SuppressWarnings("deprecation")
  public void actionPerformed(ActionEvent event) {
    if (event.getSource() == start) {
      try {
        computing_thread = new Thread() {
          public void run() {
            start.setEnabled(false);
            stop.setEnabled(true);
            do_algorithm();
            stop.setEnabled(false);
            start.setEnabled(true);
          }
        };
        computing_thread.start();
      } catch (Exception e) {
        start.setEnabled(true);
      }
    }

    if (event.getSource() == gen_problem) {
      createObjectiveFunciton();
    }

    if (event.getSource() == stop) {
      System.out.println("interrrupting");
      computing_thread.stop();
      stop.setEnabled(false);
      start.setEnabled(true);
    }
  }


  private void createObjectiveFunciton() {
    obj_func = new EvSGAAppletObjectiveFunction(40);
    problem.setContentType("text/html");
    problem.setText(obj_func.toString());
  }
}
