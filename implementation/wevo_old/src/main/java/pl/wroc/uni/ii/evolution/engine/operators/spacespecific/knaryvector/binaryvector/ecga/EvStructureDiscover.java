package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.ecga;

import java.util.Arrays;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * Computes best ECGAStructure for given population using greedySearch.
 * It tries to merge some building blocks
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 */
public class EvStructureDiscover implements
    EvOperator<EvBinaryVectorIndividual> {
  
  /**
   * Solution space used by ECGA.
   */
  private EvBinaryVectorSpace space;

  /**
   * Object storing building blocks of ECGA algorithm.
   */
  private EvECGAStructure struct;

  /**
   * input population.
   */
  private EvPopulation<EvBinaryVectorIndividual> population;

  /**
   * true - operator will try to improve building block used in previous 
   * iteration.
   * false - operator will try to discover building block from the beginning
   * during each iteration.
   */
  private boolean use_previous_structure;

  /**
   * value of log2, used frequently in algorithm.
   */
  private final double log2 = Math.log(2);


  /**
   * Constructor.
   * 
   * @param space_ binary strings solution space
   * @param use_previous_structure_
   *        <ul>
   *        <li> if <code> true </code> then discovery of the best structure
   *        uses structure from previous iteration
   *        <li> if <code> false </code> then discovery of the best structure is
   *        done from initial structure.<BR>
   *        </ul>
   */
  public EvStructureDiscover(final EvBinaryVectorSpace space_,
      final boolean use_previous_structure_) {
    this.space = space_;
    this.use_previous_structure = use_previous_structure_;
  }


  /**
   * {@inheritDoc}
   * 
   * @see pl.wroc.uni.ii.evolution.engine.prototype.EvolutionaryOperator#apply
   * 
   */
  public EvPopulation<EvBinaryVectorIndividual> apply(
      final EvPopulation<EvBinaryVectorIndividual> population_) {

    this.population = population_;
    greedySearch();

    return population;
  }


  /**
   * Trying to merge all pairs of blocks and rate each pair with their 
   * relevance to the population. <BR>
   * Only best pairs are merged (with the high rating)
   */
  private void greedySearch() {

    // creating buffer of fast Integer objects
    int pop_size = population.size();
    EvECGAFastHashMap map = new EvECGAFastHashMap(pop_size);

    /** init structure */
    if (struct == null || !use_previous_structure) {
      struct = new EvECGAStructure();

      for (int i = 0; i < space.getDimension(); i++) {
        EvBlock some = new EvBlock();
        some.put(i);

        some.setRating(getCombinedComplexity(some, population, map));
        struct.addBlock(some);
      }

    }
    /** init cache */
    EvCache merged_blocks_cache = new EvCache();

    /** iterate over all possible two elements subset */
    EvSubsetTwoIterator iterator =
        new EvSubsetTwoIterator(struct.getSetCount());
    while (iterator.hasNext()) {
      EvPair selected = iterator.next();
      EvBlock first = struct.getBlock(selected.x);
      EvBlock second = struct.getBlock(selected.y);
      EvBlock result = new EvBlock();
      result.merge(first);
      result.merge(second);

      /** compute rating for result */
      result.setRating(getCombinedComplexity(result, population, map));

      EvMergedBlock merged = new EvMergedBlock(first, second, result);
      merged_blocks_cache.put(merged);
    }

    /** do search */
    while (merged_blocks_cache.getCacheSize() > 0) {

      /** search cache */
      EvMergedBlock best_block_to_merge = merged_blocks_cache.getBestMerged();

      if ((best_block_to_merge != null)
          && (best_block_to_merge.getProfit() > 0)) {

        /** Structure update */

        /** add to structure result of best merge (a + b) */
        struct.addBlock(best_block_to_merge.getResult());
        /** remove from structure a and b */
        struct.remove(best_block_to_merge.getFirst(), best_block_to_merge
            .getSecond());

        /** Update cache */

        /** remove from cache all invalid entries */
        merged_blocks_cache.remove(best_block_to_merge.getFirst(),
            best_block_to_merge.getSecond());

        /**
         * add to cache all combination of best_block_to_merge and all other
         * blocks
         */
        for (int i = 0; i < struct.getSetCount(); i++) {
          if (struct.getBlock(i) != best_block_to_merge.getResult()) {

            EvBlock new_block = new EvBlock();
            new_block.merge(best_block_to_merge.getResult());
            new_block.merge(struct.getBlock(i));
            new_block.setRating(getCombinedComplexity(new_block, population,
                map));

            merged_blocks_cache.put(new EvMergedBlock(best_block_to_merge
                .getResult(), struct.getBlock(i), new_block));

          }
        }
      } else {
        break;
      }
    }

  }


  /**
   * Computes combined complexity for given structure of building blocks.
   * 
   * @param block - block to be rate
   * @param population_ - population used in rating the block
   * @param map - hashmap used in evaluation.
   * @return CombinedComplexity - rating of given building block
   */
  public double getCombinedComplexity(final EvBlock block,
      final EvPopulation<EvBinaryVectorIndividual> population_,
      final EvECGAFastHashMap map) {

    double cpc = getCompressedPopulationComplexity(block, population_, map);
    double mc = getModelComplexity(block, population_);

    if (Double.MAX_VALUE - mc < cpc) {
      return Double.MAX_VALUE;
    } else {
      return mc + cpc;
    }
  }


  /**
   * Computes model complexity for given block.
   * 
   * @param block - block to be rate
   * @param population_ - population used in rating the block
   * @return ModelComplexity - rating of given building block
   */
  public double getModelComplexity(final EvBlock block,
      final EvPopulation<EvBinaryVectorIndividual> population_) {

    final int maxBlockValue = 63;
    
    double mc = 0.0;
    int pop_size = population_.size();

    if (block.getSize() > maxBlockValue) {
      return Double.MAX_VALUE;
    }

    double val = Math.pow(2, block.getSize());
    if (Double.MAX_VALUE - val < mc) {
      return Double.MAX_VALUE;
    }
    mc += val;

    double logp = Math.log(pop_size) / Math.log(2);
    if (Double.MAX_VALUE / mc < logp) {
      return Double.MAX_VALUE;
    }
    return mc * logp;
  }


  /**
   * Computes compressed population complexity for given block.
   * 
   * @param block - block to be rate
   * @param population_ - population used in rating the block
   * @param map - hashmap used in evaluation.
   * @return CompressedPopulationComplexity - 
   */
  public double getCompressedPopulationComplexity(final EvBlock block,
      final EvPopulation<EvBinaryVectorIndividual> population_, 
      final EvECGAFastHashMap map) {
    // sets the Double map object
    // this helps to avoid time consuming boxing of double values
    // ########## NOTE!!! ############
    // this function is called million times and it's performance must be
    // as good as possible. Don't put any Sugar here or beautify any code or
    // put anything here without performance testing.
    // Don't worry with the code style - it's bad but because it have to run
    // fast

    double cpc = 0;

    int i = 0;
    int population_size = population.size();

    map.clear();

    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    // transforming ArrayList<Integer> to int[]
    int block_values_size = block.values.size();

    int[] block_tab = new int[block_values_size];
    for (i = 0; i < block_values_size; i++) {
      block_tab[i] = block.values.get(i);
    }
    // end of transforming ArrayList<Integer> to int[]
    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    int[] result = new int[block_values_size];

    // MAIN LOOP
    // this whole thing counts number of individuals with the same subvector
    int k;
    for (i = 0; i < population_size; i++) {

      // @@@@ function boolean[] subvector I PUT IT DIRECTLY HERE FOR BETTER
      // PERFORANCE@@@@
      EvBinaryVectorIndividual local_individual = population.get(i);

      for (k = 0; k < block_values_size; k++) {
        result[k] = local_individual.getGene(block_tab[k]);
      }
      // @@@@ end of function boolean[] subvector @@@@

      // hashing subvector so we can put it in hash table

      map.put(Arrays.hashCode(result));

    }

    double population_size_double = (double) population_size;

    int[] counts = map.getCount();

    for (i = 0; i < counts.length; i++) {
      double p = (double) counts[i] / population_size_double;
      cpc -= (Math.log(p) / log2) * p;
    }

    return cpc * population_size_double;
  }


  
  /**
   * 
   * @return current building blocks
   */
  public EvECGAStructure getStruct() {
    return this.struct;
  }

  
  /**
   * Convert structure into table of edges of size Z x 2.
   * Each block is represented by clique containing vertexes belonging to 
   * the block.
   * 
   * @return table of edges representing current structure
   */
  public int[][] getEdges() {
    return struct.getEdges();
  }
  
  
}
