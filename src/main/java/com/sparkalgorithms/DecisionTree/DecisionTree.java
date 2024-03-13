package com.sparkalgorithms.DecisionTree;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class DecisionTree {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder().appName("DecisionTree").master("local[*]").getOrCreate();
        Logger.getLogger("org.apache").setLevel(Level.WARN);
        
        Dataset<Row> dados = spark.read().option("header", true).option("inferSchema", true).csv("/home/rafael/Desktop/apple_quality.csv");

        VectorAssembler assembler = new VectorAssembler().setInputCols(new String[] {"Size", "Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", "Acidity"}).setOutputCol("features");
        
        // Converter o atributo Quality de String para int
        StringIndexer stringIndexer = new StringIndexer()
            .setInputCol("Quality")
            .setOutputCol("QualityIndex");
        dados = stringIndexer.fit(dados).transform(dados);
        
        new IndexToString() // Converter o atributo QualityIndex de int para String
            .setInputCol("QualityIndex")
            .setOutputCol("value")
            .transform(dados.select("QualityIndex").distinct());
        
        Dataset<Row> dadosFeatures = assembler.transform(dados).select("QualityIndex", "features").withColumnRenamed("QualityIndex", "label");

        DecisionTreeClassifier dtc = new DecisionTreeClassifier();
        dtc.setMaxDepth(20); // define a profundidade máxima da árvore para melhorar a precisão

        DecisionTreeClassificationModel DTmodel = dtc.fit(dadosFeatures);
        DTmodel.transform(dadosFeatures);
        
        // Definindo um novo modelo dividindo os dados em 80% para treinamento e 20% para teste
        Dataset<Row> dadosTreinamentoTotal[] = dadosFeatures.randomSplit(new double[] {0.8, 0.2});
        Dataset<Row> dadosTreinamento = dadosTreinamentoTotal[0];
        Dataset<Row> dadosTeste = dadosTreinamentoTotal[1];

        DecisionTreeClassificationModel DTmodel2 = dtc.fit(dadosTreinamento);
        Dataset<Row> previsoes = DTmodel2.transform(dadosTeste);

        previsoes.show();
        
        MulticlassClassificationEvaluator avaliador = new MulticlassClassificationEvaluator().setMetricName("accuracy"); // avalia a precisão do modelo
        System.out.println("A precisão do modelo é: " + avaliador.evaluate(previsoes)); 

        // Precisao de 0.7803398058252428
        spark.close();
    }
}