import FeatureCard from './FeatureCard';

const KeyFeatures = ({ features }) => {
  return (
    <div className="bg-white rounded-lg shadow-lg p-6 mt-6">
      <h2 className="text-xl font-bold text-gray-800 mb-4">Key Features</h2>
      <div className="grid md:grid-cols-3 gap-4">
        {Object.values(features).map(feature => (
          <FeatureCard
            key={feature.title}
            title={feature.title}
            items={feature.items}
            bgColor={feature.bgColor}
            textColor={feature.textColor}
            itemColor={feature.itemColor}
          />
        ))}
      </div>
    </div>
  );
};

export default KeyFeatures;