import { redirect } from "next/navigation";

// Insights are now merged into the Studio dashboard (#models section).
export default function InsightsPage() {
  redirect("/#models");
}
