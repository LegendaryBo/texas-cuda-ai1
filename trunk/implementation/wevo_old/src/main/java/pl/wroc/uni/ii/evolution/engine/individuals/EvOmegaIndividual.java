package pl.wroc.uni.ii.evolution.engine.individuals;

import pl.wroc.uni.ii.evolution.utils.EvRandomizer;
import java.util.ArrayList;
import java.util.Collections;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

/**
 * Class for OmegaIndividual -- a messy individual with random keys used in
 * solving permutation problems
 * 
 * @author Rafal Paliwoda (rp@message.pl)
 * @author Mateusz Malinowski (m4linka@gmail.com)
 */
public class EvOmegaIndividual extends EvMessyIndividual<Double> {

  private static final long serialVersionUID = -515091478162518078L;

  private EvOmegaIndividual template;


  /**
   * Constructor Creates new individual using given genes and alleles
   * 
   * @param expressed_genes number of expressed genes
   * @param alleles alleles
   * @param genes genes
   * @param template given template
   */
  public EvOmegaIndividual(int genotype_length, ArrayList<Integer> genes,
      ArrayList<Double> alleles, EvOmegaIndividual template) {
    super(genotype_length, genes, alleles);

    if (template != null
        && genotype_length != template.getExpressedGenesNumber()) {
      throw new IllegalArgumentException("Template has underspecified genes");
    }
    this.template = template;
  }


  /**
   * Constructor Creates new individual with full set of genes with uniformly
   * distributed random values between 0.0 and 1.0
   * 
   * @param genotype_length length of genotype
   */
  public EvOmegaIndividual(int genotype_length, EvOmegaIndividual template) {
    super(genotype_length, new ArrayList<Integer>(genotype_length),
        new ArrayList<Double>(genotype_length));

    if (template != null
        && genotype_length != template.getExpressedGenesNumber()) {
      throw new IllegalArgumentException("Template has underspecified genes");
    }

    this.template = template;

    for (int i = 0; i < genotype_length; i++) {
      genes.add(i);
      alleles.add(EvRandomizer.INSTANCE.nextDouble());
    }
  }


  /**
   * Constructor creates new individual
   * 
   * @param genotype_length genotype_length
   * @param template given template
   * @param objective_function objective function to set
   */
  public EvOmegaIndividual(int genotype_length, EvOmegaIndividual template,
      EvObjectiveFunction<EvOmegaIndividual> objective_function) {

    this(genotype_length, template);
    this.setObjectiveFunction(objective_function);
  }


  /**
   * Constructor Creates new individual with set of genes with uniformly
   * distributed random values between 0.0 and 1.0 and given length
   * 
   * @param expressed_genes number of expressed genes
   */
  public EvOmegaIndividual(int genotype_length, EvOmegaIndividual template,
      int expressed_genes) {

    this(genotype_length, template);

    while (genes.size() > expressed_genes) {
      final int j = EvRandomizer.INSTANCE.nextInt(genes.size());
      genes.remove(j);
      alleles.remove(j);
    }
  }


  /**
   * Constructor Creates new individual based on a given one
   * 
   * @param ind EvOmegaIndividual individual
   */
  public EvOmegaIndividual(EvOmegaIndividual ind) {
    super(ind.getGenotypeLength(), ind.getGenesList(), ind.getAllelesList());

    template = null;
    if (ind.template != null) {
      template = ind.getTemplate();
    }
  }


  /**
   * A template getter
   * 
   * @return template
   */
  public EvOmegaIndividual getTemplate() {
    return template;
  }


  /**
   * A template setter
   * 
   * @param template to set
   */
  public void setTemplate(EvOmegaIndividual template) {

    if (genotype_length != template.getExpressedGenesNumber())
      throw new IllegalArgumentException("Template has underspecified genes");

    this.template = template;
  }


  /**
   * Removes the template by setting it to null
   */
  public void removeTemplate() {
    template = null;
  }


  /**
   * Fills indiviual's genes up from the template
   * 
   * @return full specified individual based on this individual and the template
   */
  public EvOmegaIndividual toTemplate() {

    EvOmegaIndividual result = this.clone();
    result.removeTemplate();

    if (this.getExpressedGenesNumber() == this.getGenotypeLength()) {
      return result;
    }

    if (template == null)
      throw new IllegalStateException("Some genes are missing");

    ArrayList<Double>[] template_genotype = template.getGenotype();
    this.calculateGenotype();

    int genotype_len = genotype.length;
    for (int i = 0; i < genotype_len; i++) {
      if (genotype[i].size() == 0) {
        result.alleles.add(template_genotype[i].get(0));
        result.genes.add(i);
      }
    }

    return result;
  }


  /**
   * Returns fenotype -- a permutation It may be helpful because two individuals
   * that differ in genotype can represent the same permutation. Moreover this
   * function is used during computing number of collocation in permutation
   * represented by two individuals
   * 
   * @return fenotype
   */
  public ArrayList<Integer> getFenotype() {

    if (template == null
        && this.getExpressedGenesNumber() != this.getGenotypeLength()) {
      throw new IllegalStateException("Some genes are missing");
    }

    ArrayList<Integer> fenotype = new ArrayList<Integer>(genotype_length);
    ArrayList<Double> simplified_genotype =
        new ArrayList<Double>(genotype_length);
    ArrayList<Double> sorted = new ArrayList<Double>();

    this.calculateGenotype();
    int genotype_len = genotype.length;
    for (int i = 0; i < genotype_len; i++) {
      if (genotype[i].size() > 0) {
        simplified_genotype.add(genotype[i].get(0));
      } else {
        simplified_genotype.add(null);
      }
    }

    ArrayList<Double>[] template_genotype = null;
    if (template != null) {
      template_genotype = template.getGenotype();
    }
    // collecting missing genes from template
    for (int i = 0; i < genotype_length; i++) {
      if (simplified_genotype.get(i) == null)
        simplified_genotype.set(i, template_genotype[i].get(0));
    }

    for (int i = 0; i < genotype.length; i++) {
      sorted.add(simplified_genotype.get(i));
    }

    // mapping random keys into permutation
    // by arranging it in ascending order
    // and reading old place of each allele

    ArrayList<Boolean> table = new ArrayList<Boolean>(genotype.length);
    for (int i = 0; i < genotype.length; i++) {
      table.add(false);
    }

    Collections.sort(sorted);
    int sorted_size = sorted.size();

    for (int i = 0; i < sorted_size; i++) {
      for (int j = 0; j < sorted_size; j++) {
        if (simplified_genotype.get(j) == sorted.get(i) && !table.get(j)) {
          fenotype.add(j);
          table.set(j, true);
          break;
        }
      }
    }
    return fenotype;
  }


  @Override
  public int compareTo(Object o) {
    EvOmegaIndividual individual = (EvOmegaIndividual) o;
    if (individual.getObjectiveFunctionValue() > getObjectiveFunctionValue()) {
      return -1;
    } else if (individual.getObjectiveFunctionValue() < getObjectiveFunctionValue()) {
      return 1;
    } else
      return 0;
  }


  @Override
  public EvOmegaIndividual clone() {

    EvOmegaIndividual cloned =
        new EvOmegaIndividual(genotype_length, new ArrayList<Integer>(genes),
            new ArrayList<Double>(alleles), template);

    for (int i = 0; i < this.getObjectiveFunctions().size(); i++) {
      cloned.addObjectiveFunction(this.getObjectiveFunction(i));
      if (this.isEvaluated(i)) {
        cloned.assignObjectiveFunctionValue(getObjectiveFunctionValue(i), i);
      }
    }

    return cloned;
  }
}
