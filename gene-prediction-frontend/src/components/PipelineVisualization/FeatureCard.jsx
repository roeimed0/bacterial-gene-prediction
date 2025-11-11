const FeatureCard = ({ title, items, bgColor, textColor, itemColor }) => {
  return (
    <div className={`p-4 ${bgColor} rounded-lg`}>
      <h3 className={`font-semibold ${textColor} mb-2`}>{title}</h3>
      <ul className={`text-sm ${itemColor} space-y-1`}>
        {items.map((item, idx) => (
          <li key={idx}>• {item}</li>
        ))}
      </ul>
    </div>
  );
};

export default FeatureCard;